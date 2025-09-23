
import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv(override=True)

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
# from langchain.agents import create_agent  # its deprecated now
from langchain.agents import AgentExecutor, create_tool_calling_agent
 

# initiate the LLM - OpenSource Model(from Groq)
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0) # model="llama-3.1-8b-instant"

### Tool 1 - Tavily Search from langchain

os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

from langchain_tavily import TavilySearch

tavily_search_tool = TavilySearch(
    max_results = 5,
    topic = "general"
)


### Tool 2 - Semantic Search from langchain
from langchain_community.tools.semanticscholar.tool import SemanticScholarQueryRun
from langchain_community.utilities.semanticscholar import SemanticScholarAPIWrapper
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain import hub

instructions = """You are an expert researcher."""
base_prompt = hub.pull("langchain-ai/openai-functions-template")
prompt = base_prompt.partial(instructions=instructions)

# Initialize Semantic Scholar Search
tools_semantic_search = [SemanticScholarQueryRun()]
agent = create_tool_calling_agent(llm, tools_semantic_search, prompt)

agent_semantic = AgentExecutor(
    agent = agent,
    tools = tools_semantic_search,
    verbose=True
)


### Tool 3 - Youtube url search

from langchain_community.tools import YouTubeSearchTool
youtube_tool = YouTubeSearchTool()



from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display

# Graph State
class State(TypedDict):
    topic: str
    summary: str


# Nodes
def collect_and_explain_resources(state: State):
    collected_data = ""


    # semantic scholar search
    semantic_query = f"query: basis on topic {state['topic']}, create 3 selections: paper name with URL, summary and citations. "
    print("searching for semantic results")
    semantic_agent_response = agent_semantic.invoke({"input": semantic_query}) # invoke the agent function here
    print("got semantic results")
    collected_data += f"### Semantic Scholar Results: \n{semantic_agent_response}\n\n "


    # YouTube Search
    print("searching for youtube results")
    youtube_results = youtube_tool.run(f"{state['topic']}, 3") # get top 3 youtube results
    print("got youtube results")
    collected_data += f"### YouTube Results :\n {youtube_results} \n\n"


    # Tavily Search
    print("searching for Tavily Search")
    web_search_query = f"find 3 articles/blogs on {state['topic']} and get back top 3 articles/blogs with their names and URL for each. "
    web_results = tavily_search_tool.invoke({"query": web_search_query})
    print("got the Tavily Search result")
    collected_data += f"### Web Articles and Blogs: \n{web_results} \n\n"


    prompt_search = ChatPromptTemplate.from_messages(
        [
            ("system",
            """ You are professional research expert and you have deep experience in any research topic. The User will give you a topic to research. 
            Your job is to analyze the topic using tools and give summary as instructed below, 
            1. You will use Semantic Scholar Search to get paper name with URL, summary and citation 
            2. You will use YouTube Tool to get top 3 url's which are relevant to topic 
            3. You will use Tavily Search to get the details from internet and return top 3 blogs/articals with their names and URL for each

            Based on above, you will summarize the information sequetially by followiing pointers 1, 2 and 3.  

            Here is the data to organize {collected_data}\n
            Please ensure that final output is well structured and easy to understand
          
            """
            ),
            ("user","Get the final structured output")
            ])

    # Format the prompt with collected data
    formatted_output = prompt_search.format_prompt(collected_data=collected_data)
    # generate the response
    structured_output = llm.invoke(formatted_output.to_messages())

    # if structured output is a list or has multiple messages
    if isinstance(structured_output, list):
        final_text = structured_output[0].content
    else:
        final_text = getattr(structured_output, "content", str(structured_output))
    
    return {"summary": final_text}



# add State Graph
graph_builder = StateGraph(State)

# add Nodes
graph_builder.add_node("collect_and_explain_resources", collect_and_explain_resources)

# add edges
graph_builder.add_edge(START, "collect_and_explain_resources")
graph_builder.add_edge("collect_and_explain_resources", END)

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
                st.write(structured_result["summary"])
        else:
            st.warning("Please enter a topic to collect resources.")

