### report_analysis.py

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import textwrap

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import os
from dotenv import load_dotenv
load_dotenv(override=True)

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


# initialize the LLM
# llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# analyse the report thoroughly and try to understand

def report_analysis(df: pd.DataFrame):
    # convert small summary of data to string
    data_preview = df.head(10).to_markdown(index=False)
    schema = "\n".join([f" - {col}: {dtype}" for col, dtype in df.dtypes.items()])


        
    prompt_analyse = ChatPromptTemplate.from_messages([
        ("system",
        f""" You are an expert in report analysing task. You have 20 years of experience in reporting. You will be provided dataset in csv file. You will read the file carefully and try to understand it. 
        Take your time to understand it correctly. Then you will be expected to generate below outputs,
        1. Summarize the report and describe what it is about.
        2. Generate 4-5 graphs which helps client to understand the report visually.
        3. Provide 4-5 insights from report
        4. Suggest improvements if applicable.

        ### Important Instructions:
        - When creating plots, always use `hue` explicitly if you set a `palette`.
        - Do NOT use `plt.show()`; instead, let the figure be returned so Streamlit can display it.
        - Generate 4-5 clear and visually helpful graphs.

        
        Use markdown formatting in your output.

        **schema:**
        {schema}

        **Preview:**
        {data_preview}

        Output in markdown. Put each plot in proper Python code blocks using triple backticks.
        Do not use import statements in code.
        """
        ),
        ("user","Here is the dataset. Please analyze it and follow all the instructions. ")
    ])

    
    # combine the model
    chain_analyse = prompt_analyse | llm

    result_analyse = chain_analyse.invoke({})

    return result_analyse.content





"""
This function takes a string (response_text) as input.
The goal is to find Python code blocks in the text, run them, and collect any plots (figures) that the code generates.
"""

def extract_and_plot_code_blocks(response_text):
    import re

    code_blocks = re.findall(r"```python(.*?)```", response_text, re.DOTALL)
    figs = []

    """
    It uses a regular expression (re.findall) to search for text between:
    re.DOTALL makes sure that even multi-line code blocks are captured.
    Result:
    code_blocks will be a list of code snippets.
    """

    for code in code_blocks:    # It goes through each code snippet found.
        local_vars = {}         # local_vars is a dictionary to store any variables created when running the code.

        # clean indentation
        cleaned_code = textwrap.dedent(code).strip()  # Removes any extra indentation and unnecessary spaces.

        code_to_exec = f""" 
df = st.session_state.df.copy()     # Before running the user's code, it automatically creates a copy of a dataframe (df) stored in st.session_state.df.
        

{cleaned_code} 
"""

        try:
            exec(code_to_exec, {'plt': plt, 'sns': sns, 'pd': pd, 'st': st}, local_vars)    # exec() runs the code dynamically like Python would run it in a script.

            # captures all open figures
            for fig_num in plt.get_fignums():   # After the code runs, this finds all the active Matplotlib figures that were created.
                fig = plt.figure(fig_num)
                figs.append(fig)

            # close all figures so they dont overlap
            plt.close(fig)      # After capturing a figure, it closes it, so it doesn’t overlap with the next plot.

                #fig = plt.gcf()
            #figs.append(fig)
            #plt.clf()
        except Exception as e:
            st.warning(f"Error while executing generated code:\n{e}")

    return figs






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




    st.title("GPT-Powered CSV analyser Report")

    uploaded_file = st.file_uploader("Upload a csv file: ", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df

        with st.spinner("Anaylyzing CSV and generating visualization..."):
            response_text = report_analysis(df)

        st.subheader(" Report Summary + Insights")
        sections = response_text.split("```python")
        st.markdown(sections[0]) # everything before code

        st.subheader(" Generated Graphs")


        figs = extract_and_plot_code_blocks(response_text)
        if figs:
            for i, fig in enumerate(figs, 1):
                st.subheader(f"Generated Graph {i}")
                st.pyplot(fig)
        else:
            st.warning("No graphs were generated by the AI.")


        """
        figs = extract_and_plot_code_blocks(response_text)
        for fig in figs:
            st.pyplot(fig)
        """
        
        
        
        st.success(" Analysis Completed!")

        del st.session_state.df
