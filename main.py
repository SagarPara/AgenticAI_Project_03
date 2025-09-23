### main.py

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
