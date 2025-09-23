### fun_with_grammar.py

import os
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


load_dotenv(override=True)

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# initialize the LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

def generate_grammar_exercise():
    """
    Generates a single English grammar exercise question at a time
    suitable for students from 5th to 8th grade.
    """
    # create a chat prompt template
    prompt_system = ChatPromptTemplate.from_messages([
            (
                "system", "You are a English Teacher. "
                "Your job is to teach people english grammar, via fun and interesteting short exercises. "
                "Provide one question at a time, either fill-in-blanks or multiple-choice. "
                "please note that while generating questions for english grammar, treat yourself as 5th to 8th standard teacher."
                "Remember you should generate new question on click on Start button everytime "
            ),
            (
                "user", 
                "please generate a single english grammar exercise question on click on start button. Remember you should generate new question everytime on click of start button "
            )
        ])


    # combine prompt and llm model
    chain = prompt_system | llm

    # run the chain
    # response = chain.invoke({"question_type": "fill-in-the-blank"})
    response = chain.invoke({})

    # print the output
    print("Generated Question: ")
    # print(response.content)
    return response.content


def check_answer(question, user_answer):
    """
    Evaluate the user's answer and give feedback based on answer provided by user.
    """

    prompt_check_answer = ChatPromptTemplate.from_messages([
        (
            "system", "You are a English language teacher with quite good experience. Your job is to teach english language to students. "
            "You will be given a question and it's answer to evaluate, both by the User. "
            "You will be evaluating it and share feedback. Please be supportive and helpful. "
        ),
        (
            "user", f"Question: {question} \n Answer: {user_answer} \n" 
            "Evaluate the corretness of the answer and provide feedback. "
        )
    ])

    chain_answer = prompt_check_answer | llm

    response_answer = chain_answer.invoke({})

    # print the output
    print("\n Feedback")
    # print(response_answer.content)
    return response_answer.content



### app.py :--> Streamlit UI

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



    st.title("Fun With Grammar")
    st.write(" Sharpen your grammar skills with these exercises.")

    # State Management for exercise generation and user input
    if "exercise" not in st.session_state:
        st.session_state.exercise = None
    if "user_response" not in st.session_state:
        st.session_state.user_response = ""
    if "feedback" not in st.session_state:
        st.session_state.feedback = None


    # build exercise button
    if st.button("Start"):
        st.session_state.exercise = generate_grammar_exercise()
        st.session_state.user_response = ""
        st.session_state.feedback = None

    if st.session_state.exercise:
        st.subheader("Exercise")
        st.write(st.session_state.exercise)

        # user input
        user_response = st.text_input("Your answer :", key = "response")
        

        if st.button("Check Answer"):
            if user_response:
                st.session_state.user_response = user_response
                feedback = check_answer(st.session_state.exercise, user_response)
                st.session_state.feedback = feedback
            else:
                st.error("Please enter an answer before checking")


        
        # show feddback if available
        if st.session_state.feedback:
            st.subheader("Feedback on Your Answer")
            st.write(st.session_state.feedback)

            
