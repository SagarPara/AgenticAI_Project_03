### home.py
def app():
    # head.py  (Home / Landing page)
    import streamlit as st
    import uuid

    # Page config (ok to call on each page; settings should be identical across pages)
    st.set_page_config(
        page_title="AI Research Assistant",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # ---------------------------------------------------------------------
    # CSS: dark theme + hide Streamlit's built-in pages navigation (duplicate)
    # ---------------------------------------------------------------------
    # NOTE: the selectors target Streamlit's page nav in the sidebar.
    # If Streamlit changes DOM in future, adjust selectors accordingly.
    CUSTOM_CSS = """
    <style>
    /* App background and text */
    [data-testid="stAppViewContainer"], .main, .block-container {
    background-color: #000000 !important;
    color: #FFFFFF !important;
    }

    /* Main content headings & paragraphs */
    h1, h2, h3, h4, h5, h6 { color: #FFFFFF !important; }
    .stMarkdown p, .stText { color: #E6E6E6 !important; }

    /* Sidebar background */
    [data-testid="stSidebar"] {
    background-color: #0b0b0b !important;
    color: #E6E6E6 !important;
    }

    /* Hide Streamlit automatic Pages navigation that appears in the sidebar
    This prevents the duplicate top-left menu you mentioned. */
    section[data-testid="stSidebarNav"] { display: none !important; }

    /* Optional: hide the Streamlit top-left "menu" area if present */
    header[role="banner"] { background-color: #000000 !important; }

    /* Buttons look */
    .stButton>button {
    background-color: #1f6feb !important;
    color: #ffffff !important;
    border-radius: 6px;
    padding: 8px 10px;
    }

    /* Input boxes (dark-looking) */
    [data-testid="stTextInput"] input, .stTextInput>div>div>input {
    background-color: #121212 !important;
    color: #ffffff !important;
    border: 1px solid #222 !important;
    }

    /* Make code blocks easy to read on dark bg */
    pre, code {
    background-color: #0b0b0b !important;
    color: #e6e6e6 !important;
    }
    </style>
    """
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # --------------------------------------
    # Page content
    # --------------------------------------
    st.title("Welcome to Your AI Research Assistant")
    st.subheader("Your AI-powered companion for accelerating and simplifying your research journey.")
    st.write(
        """
        This assistant helps researchers, students, and professionals find, summarise, and organise
        research materials quickly. Use the left sidebar to navigate to feature pages:
        - Fun With Grammar
        - Language Translation
        - Report Analysis
        - Research Collection
        - Assistance ChatBot
        """
    )

    st.markdown("---")

    
    st.header("Capabilities of the AI Research Assistant")
    st.markdown(
            """
            **1. Speed and Efficiency**  
            Rapidly generate ideas, collect papers, articles and videos, and summarize them so you can move
            straight to analysis.

            **2. Simplification of Complex Tasks**  
            The assistant breaks down research workflows — ideation, collection, summarization — into manageable steps.

            **3. AI-Powered Insights**  
            The assistant helps identify trends, gaps, and related concepts that may not be immediately apparent.

            **4. Integrated Tools**  
            - Academic search (Semantic Scholar)
            - Web/article search (Tavily)
            - YouTube discovery
            - Document RAG via uploaded PDFs (your resume, research notes)
            - Small utilities (math ops) exposed to the agent
            """
        )

    st.markdown("### How it works")
    st.write(
            """
            - **Ideating:** Use the Ideation page to brainstorm and expand research questions.  
            - **Collecting:** Use Research Collection to fetch and organize papers, blogs, and videos.  
            - **Analyzing:** Upload CSVs on Report Analysis to generate plots and insights.  
            - **Chat:** Use Assistance ChatBot to ask questions (it can call retrievers over your uploaded PDFs).
            """
        )

    st.markdown("---")
    st.caption("Want this page styled differently? Edit `head.py` (this file) or the shared CSS block at the top.")

    # Optional: small footer
    st.write("")
    st.markdown("© AI Research Assistant • Built by Sagar Parab" , unsafe_allow_html=True)
