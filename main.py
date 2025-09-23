### main.py
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Now safely set environment variables (with check)
required_keys = ["OPENAI_API_KEY", "GROQ_API_KEY", "TAVILY_API_KEY"]

for key in required_keys:
    value = os.getenv(key)
    if not value:
        raise ValueError(f"{key} is not set in the environment!")
    os.environ[key] = value
    


import streamlit as st
from pages import home, fun_with_grammar, language_translation, report_analysis, research_collection, chatbot

# Hide default Streamlit pages menu
st.markdown("""
    <style>
        section[data-testid="stSidebarNav"] {display: none;}
    </style>
""", unsafe_allow_html=True)

PAGES = {
    "home": home,
    "Fun With Grammar ": fun_with_grammar,
    "Language Translation": language_translation,
    "Report Analysis": report_analysis,
    "Research Collection": research_collection,
    "Assistance ChatBot" : chatbot
}

def main():

    
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))

    page = PAGES[selection]
    page.app()
    # page()

if __name__ == "__main__":
    main()
