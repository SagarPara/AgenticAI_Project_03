### language_translation.py


import streamlit as st
import os
import re
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate

load_dotenv(override=True)

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# used open-source model from Alibaba Cloud [google open source model --> gemma2-9b-it]
llm = ChatGroq(model="qwen/qwen3-32b")

def generate_hindi_sentence(difficulty):
    """
    Generates a long hindi sentence
    """
    # create a chat prompt template
    prompt_system = ChatPromptTemplate.from_messages([
        (
            "system",
            f"You are a professional Hindi language expert and teacher. You are assisting in creating Hindi language exercises for translation purposes based on difficulty level."
            "You should only respond with one single sentence in Hindi. The sentence should be between 80 and 100 words long, and grammatically correct. "
            "Do not explain your reasoning. Do not output anything except the sentence. Do not include any internal thoughts, tags, or formatting like <think>."
            "The difficulty level is: {difficulty}."
        ),
        (
            "user",
            f"Act like a content generator for language learners and generate a {difficulty.lower()} level of hindi sentence for learners. Only provide the Hindi sentence. No other output."
        )
        ])

    chain_system = prompt_system | llm

    response_system = chain_system.invoke({"difficulty":difficulty})
    raw_output = response_system .content

    # Strip out <think> and anything inside it
    clean_output = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL).strip()


    print("generating hindi_sentence")
    return clean_output


def evaluating_hindi_sentence(origional, translation):
    prompt_check = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are language teacher with 20 years of experience. Your job is to check the translation done by the user from hindi to english."
            "you will be given origional sentence and translation done by user ."
            "you have to point out what user did wrong in answer, and what can be improved. if everthing is fine then you should appreciate the user. "
            "Do not include any internal thoughts, tags, or formatting like <think>."
        ),
        (
            "user", 
            f"origional sentence in Hindi: {origional}, /n user translation: {translation}."
          "Based on the origional sentence and user generated answer, you will evaluate the answer and provide feedback about what is going wrong and right "
          "Do not include any internal thoughts, tags, or formatting like <think>."
          )
          ])

    chain_check = prompt_check | llm

    response_check = chain_check.invoke({})
    raw_output_check = response_check.content

    # Strip out <think> and anything inside it
    clean_output_check = re.sub(r"<think>.*?</think>", "", raw_output_check, flags=re.DOTALL).strip()

    print("Generating Evaluation answer")
    return clean_output_check


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


    st.title("Hindi to English Language Translation :")
    st.write(" Enhance your language skills with these exercises.")

    difficulty = st.selectbox("Choose difficulty level:", ["Easy", "Medium", "Hard"])


    # State Management for exercise generation and user input
    if "generated_sentence" not in st.session_state:
        st.session_state.generated_sentence = None
    if "translation_input" not in st.session_state:
        st.session_state.translation_input = ""
    if "feedback_system" not in st.session_state:
        st.session_state.feedback_system = None


    # build exercise button
    if st.button("Start"):
        st.session_state.generated_sentence = generate_hindi_sentence(difficulty)
        st.session_state.translation_input = ""
        st.session_state.feedback_system = None

    if st.session_state.generated_sentence:
        st.subheader("Hindi sentence to translate :")
        st.write(st.session_state.generated_sentence)

        # user input for translation
        user_translation = st.text_input("Your translation :", key = "translation")
        

        if st.button("Verify Answer"):
            if user_translation:
                st.session_state.translation_input = user_translation
                correction = evaluating_hindi_sentence(st.session_state.generated_sentence, user_translation)
                st.subheader("Translation Feedback: ")
                st.write(correction)
            else:
                st.error("Please enter an translation before checking")


        """
        # show feddback if available
        if st.session_state.feedback:
            st.subheader("Feedback on Your Answer")
            st.write(st.session_state.feedback)

        
        """
            
