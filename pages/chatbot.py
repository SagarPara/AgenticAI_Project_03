# chatbot.py

### --- Agentic RAG ---
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from typing import Annotated
#from langchain_core.messages import BaseMessage
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langgraph.prebuilt import tools_condition
from langgraph.checkpoint.memory import MemorySaver


import os
from dotenv import load_dotenv
load_dotenv(override=True)


os.environ["OPENAI_API_KEY"] == os.getenv("OPENAI_API_KEY")

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

pdf_files_1 = [r"C:\Users\Dell\OneDrive\Documents\Sagar_Parab_Resume.pdf"]
             

docs_1 = [PyPDFLoader(doc).load() for doc in pdf_files_1]
print(docs_1)

docs_list_1 = [item for sublist in docs_1 for item in sublist]

text_splitter_1 = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

doc_splits_1 = text_splitter_1.split_documents(docs_list_1)
## add all these text to vectordb
vectorstore_1 = FAISS.from_documents(
    documents=doc_splits_1,
    embedding=OpenAIEmbeddings()
)

retriever_1 = vectorstore_1.as_retriever()

### retriver tool
retriever_resume_tool_1 = create_retriever_tool(
    retriever_1,
    "retriever_vector_db_blog",
    "search and find the information about Sagar resume"
)

pdf_files_2 = [r"C:\Users\Dell\OneDrive\Documents\BERT summary.pdf"]

docs_2 = [PyPDFLoader(doc).load() for doc in pdf_files_2]
print(docs_2)

docs_list_2 = [item for sublist in docs_2 for item in sublist]

text_splitter_2 = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=50)

doc_splits_2 = text_splitter_2.split_documents(docs_list_2)
## add all these text to vectordb
vectorstore_2 = FAISS.from_documents(
    documents=doc_splits_2,
    embedding=OpenAIEmbeddings()
)

retriever_2 = vectorstore_2.as_retriever()

### retriver tool
retriever_resume_tool_2 = create_retriever_tool(
    retriever_2,
    "retriever_vector_db1_blog",
    "search and find the information about BERT if ask regarding BERT model or BERT"
)



# create State class
class State(TypedDict):
    messages:Annotated[list, add_messages]

### --- built up tools ---
def add(a:int, b:int)-> int:
    """ Add a and b
    Args:
        a (int): first int
        b (int): second int

    Returns:
        int
    """
    return a+b


def substract(a:int, b:int)-> int:
    """ Substract a and b
    Args:
        a (int): first int
        b (int): second int

    Returns:
        int
    """
    return a-b

def multiply(a:int, b:int)-> int:
    """ Multiply a and b
    Args:
        a (int): first int
        b (int): second int

    Returns:
        int
    """
    return a*b

def divide(a:int, b:int)-> int:
    """ Divide a and b
    Args:
        a (int): first int
        b (int): second int

    Returns:
        int
    """
    return a/b


tools = [add, substract, multiply, divide, retriever_resume_tool_1, retriever_resume_tool_2]

def llm_agent(state: State):
    """
    Invokes this model to generate the response based on the current state given.
    Given the question, model will decide to retrieve using the retriever tool, or simply end.

    Args:
        state (messages): The current state
    Returns:
        dict: The updated state with the agent response appended to messages
    """
    print("---CALL AGENT---")
    messages = state["messages"]
    llm_with_tools = llm.bind_tools(tools)
    respose = llm_with_tools.invoke(messages)

    # we will return a list, because this will get added to the existing list
    return {"messages": [respose]}


graph_builder = StateGraph(State)

### node
graph_builder.add_node("llm_agent", llm_agent)
graph_builder.add_node("retrieve", ToolNode(tools))


graph_builder.add_edge(START, "llm_agent")
graph_builder.add_conditional_edges("llm_agent", tools_condition, {"tools":"retrieve", END:END})
graph_builder.add_edge("llm_agent", END)


memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

### Display
from IPython.display import Image, display

# graph_image = graph.get_graph().draw_mermaid_png()
# display(Image(graph_image))


# To make unique per User, generating the thread ID dynamically
import uuid
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())


# Step 6: Streamlit App
# ---------------------------
import streamlit as st
import uuid

def app():
    
    ### background screen in black color
    st.set_page_config(
        page_title="Language Tutor",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Force dark theme
    st.markdown(
        """
        <style>
            body {
                background-color: black;
                color: white;
            }
        </style>
        """,
        unsafe_allow_html=True
    )



    st.title("🤖 Agentic RAG Chatbot")
    st.write("Ask general questions or specifically about Sagar's resume or BERT summary.")

    # Initialize session state variables
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())  # unique ID per session

    # --- Display chat messages like a conversation ---
    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]):  # 'user' or 'assistant'
            st.markdown(chat["content"])

    # --- Chat input box at the bottom ---
    user_input = st.chat_input("Type your message here...")

    if user_input:
        # Save user message
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # Display immediately
        with st.chat_message("user"):
            st.markdown(user_input)

        # Prepare state for LangGraph
        state = {"messages": [{"role": "user", "content": user_input}]}

        # Run graph
        result = graph.invoke(
            state,
            config={
                "configurable": {
                    "thread_id": st.session_state.thread_id
                }
            }
        )

        # Extract LLM response
        ai_message = (
            result["messages"][-1].content if result.get("messages") else "No response"
        )

        # Save bot response
        st.session_state.chat_history.append({"role": "assistant", "content": ai_message})

        # Display bot message immediately
        with st.chat_message("assistant"):
            st.markdown(ai_message)

