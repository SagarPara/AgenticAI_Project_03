
import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv(override=True)

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
# from langchain.agents import create_agent  # its deprecated now
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display



# initiate the LLM - OpenSource Model(from Groq)
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0) # model="llama-3.1-8b-instant"


# create State class
class State(TypedDict):
    topic: str
    tavily_result: list
    youtube_result: list
    semantic_result: list
    final_result: str


### --- Create Langgraph Nodes ---

### 1. ToolNode - Tavily Search from langchain
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
from langchain_tavily import TavilySearch


def tavily_search_node(state: State):
    tavily_result = []
    tavily_query_prompt = (
        f"find top 3 blogs/articals with their names and URL for each on the topic: {state['topic']}.\n"
        "For each paper return:\n"
        "1. Paper Title \n"
        "2. URL \n"
        "3. Summary (max 2 sentences) \n "
    )

    print("browsing for Tavily Search query")
    tavily_tool = TavilySearch(max_results = 5)
    raw_result = tavily_tool.run(tavily_query_prompt) 

    # Clean the output
    tavily_result = []
    if "results" in raw_result:
        for item in raw_result["results"][:3]:  # limit to top 3
            tavily_result.append({
                "title": item.get("title", "No title"),
                "url": item.get("url", "No URL"),
                "summary": item.get("content", "No summary")
            })



    return {"tavily_result": tavily_result}


### 2. ToolNode - Semantic Search
from langchain_community.tools.semanticscholar.tool import SemanticScholarQueryRun
from langchain_community.utilities.semanticscholar import SemanticScholarAPIWrapper
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain import hub

instructions = """You are an expert researcher."""
base_prompt = hub.pull("langchain-ai/openai-functions-template")
prompt = base_prompt.partial(instructions=instructions)

def semantic_search_node(state: State):
    semantic_result = []
    semantic_query_prompt = f"""
    Find 2 paper names with URL, summary and citation for each on the topic: {state['topic']}.
    For each paper return:
    1. Paper Title
    2. URL
    3. Summary (max 2 sentences)
    4. Citation count
    """   
    

    print("browsing for Semantic Search query...")
    raw_result = SemanticScholarQueryRun().run(semantic_query_prompt)

    # Convert text output into list of dictionaries
    if isinstance(raw_result, str):
        semantic_result = [{"result": raw_result}] if raw_result else []
    else:
        semantic_result = raw_result



    return {"semantic_result": semantic_result}


    
### Tool 3 - Youtube url search
from langchain_community.tools import YouTubeSearchTool
def youtube_search_node(state: State):
    youtube_result = []
    youtube_query_prompt = (
        f"find top 3 URL links which are relevant to the topic: {state['topic']} .\n"
    )

    print("browsing for youtube search query")
    raw_result = YouTubeSearchTool().run(youtube_query_prompt)
    

    # Convert string to list if needed
    if isinstance(raw_result, str):
        try:
            raw_result = eval(raw_result)  # convert string list to actual list
        except:
            raw_result = [raw_result]

    youtube_result = [{"url": url} for url in raw_result[:3]]


    return {"youtube_result": youtube_result}



def combine_elements(state: State):
    final_prompt = (
        f"write a organize and summarize research analysis using these elements "
        f"Tavily_Result: {state['tavily_result']}\n"
        f"Semantic_Result: {state['semantic_result']}\n"
        f"YouTube_Result: {state['youtube_result']}"

    )

    final_result = llm.invoke(final_prompt)

    return {"final_result": final_result.content}
    

 


# add State Graph
graph_builder = StateGraph(State)

# add Nodes
graph_builder.add_node("semantic_search_node", semantic_search_node)
graph_builder.add_node("tavily_search_node", tavily_search_node)
graph_builder.add_node("youtube_search_node", youtube_search_node)
graph_builder.add_node("combine_elements", combine_elements)

# add edges
graph_builder.add_edge(START, "semantic_search_node")
graph_builder.add_edge(START, "tavily_search_node")
graph_builder.add_edge(START, "youtube_search_node")
graph_builder.add_edge("semantic_search_node", "combine_elements")
graph_builder.add_edge("tavily_search_node", "combine_elements")
graph_builder.add_edge("youtube_search_node", "combine_elements")
graph_builder.add_edge("combine_elements", END)

# compile and run
graph = graph_builder.compile()
# graph_image = graph.get_graph().draw_mermaid_png()
# display(Image(graph_image))


### Streamlit UI
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



    st.title("Research Collection: Gather and Organize Resources")
    st.subheader("Effortlessly collect and organize research materials.")

    st.write("""
    Input a specific topic, question, or area of interest, and the AI will retrieve relevant academic papers, 
    articles, YouTube videos, and blogs. The assistant will organize these resources for easier review and synthesis.
    """)

    topic = st.text_area("Enter your research topic :")

    if st.button("Collect Resources"):
        if topic:
            with st.spinner("Collecting and organizing resources..."):
                state = {"topic":topic}
                structured_result = graph.invoke(state)
                st.write("### Structured Research Summary")
                st.write(structured_result)
        else:
            st.warning("Please enter a topic to collect resources.")

