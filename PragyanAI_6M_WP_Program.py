import streamlit as st
import pandas as pd
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- Page Configuration ---
st.set_page_config(
    page_title="PragyanAI - Executive Program in Generative & Agentic AI",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom CSS for Styling ---
st.markdown("""
<style>
    /* General Styles */
    .stApp {
        background-color: #0f172a;
        color: #cbd5e1;
    }
    h1, h2, h3 {
        color: #ffffff;
        font-weight: 700;
    }
    .stButton>button, .stDownloadButton>button {
        background-color: #f97316;
        color: white;
        border-radius: 0.5rem;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        border: none;
        transition: background-color 0.3s;
    }
    .stButton>button:hover, .stDownloadButton>button:hover {
        background-color: #ea580c;
    }
    
    .stLinkButton>a {
        background-color: #ffffff;
        color: #f97316 !important;
        border-radius: 0.5rem;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        border: 1px solid #f97316;
        transition: background-color 0.3s, color 0.3s;
        text-decoration: none;
    }
    .stLinkButton>a:hover {
        background-color: #f1f5f9;
        color: #ea580c !important;
    }

    /* Custom Classes */
    .gradient-text {
        background: linear-gradient(to right, #f97316, #ea580c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .section-header {
        text-align: center;
        margin-bottom: 2rem;
    }
    .highlight-card {
        background-color: #1e293b;
        padding: 1.5rem;
        border-radius: 1rem;
        border: 1px solid #334155;
        text-align: center;
        height: 100%;
    }
    .footer {
        border-top: 1px solid #334155;
        padding-top: 2rem;
        margin-top: 4rem;
        text-align: center;
        color: #94a3b8;
    }
</style>
""", unsafe_allow_html=True)


# --- Data Loading Function ---
@st.cache_data(ttl=600) # Cache data for 10 minutes
def load_data(sheet_url):
    """Loads Q&A data from a public Google Sheet."""
    try:
        csv_url = sheet_url.replace("/edit#gid=", "/export?format=csv&gid=")
        df = pd.read_csv(csv_url)
        # Ensure the columns are named correctly, even if they are not in the sheet
        if 'Question' not in df.columns or 'Answer' not in df.columns:
            st.error("Google Sheet must contain 'Question' and 'Answer' columns.")
            return None
        return df
    except Exception as e:
        st.error(f"Failed to load data from Google Sheet: {e}")
        return None

# --- Header ---
col1, col2, col3 = st.columns([2, 5, 2])
with col2:
    st.image("PragyanAI_Transperent_github.png", use_container_width=True)

# --- Hero Section ---
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<h1 style="text-align: center;">Executive Program in <span class="gradient-text">Generative & Agentic AI</span></h1>', unsafe_allow_html=True)
st.markdown('<h3 style="text-align: center; color: #94a3b8;">A 6-Month Weekend Accelerator for Senior IT Professionals (4+ Years Experience)</h3>', unsafe_allow_html=True)
st.markdown("""
<p style="text-align: center; max-width: 700px; margin: auto; padding-top: 1rem;">
    In a tech landscape where AI is actively reshaping job roles, proactive upskilling is essential. This program is a decisive move to future-proof your career and transition from an IT professional to an <strong>AI Solution Architect and Leader.</strong>
</p>
""", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)


# --- Program Highlights ---
st.markdown('<h2 class="section-header">Program Highlights</h2>', unsafe_allow_html=True)
cols = st.columns(4)
highlights = {
    "Duration": "6 Months",
    "Schedule": "Sat & Sun (Live)",
    "Audience": "IT Professionals (4+ Yrs)",
    "Fee": "₹1,00,000"
}
for i, (title, value) in enumerate(highlights.items()):
    with cols[i]:
        st.markdown(f'<div class="highlight-card"><h3>{title}</h3><p style="font-size: 1.5rem; color: #f97316; font-weight: 600;">{value}</p></div>', unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

# --- Curriculum Section ---
st.markdown('<h2 class="section-header">Mastery Across 8 Core AI Domains</h2>', unsafe_allow_html=True)
domains = [
    "1. Advanced Python & Data Engineering", "2. Applied Data Science & Analytics",
    "3. BI & Big Data Architecture", "4. Practical Machine Learning & MLOps",
    "5. Applied Deep Learning & Computer Vision", "6. NLP & Conversational AI",
    "7. Generative AI & Large Language Models", "8. Agentic AI & MVP Development"
]
rows = [st.columns(4) for _ in range(2)]
flat_list = [item for sublist in rows for item in sublist]
for i, domain in enumerate(domains):
    with flat_list[i]:
        st.info(domain)

st.markdown("<br><br>", unsafe_allow_html=True)

# --- Q&A Section ---
st.markdown('<h2 class="section-header">Have Questions? Ask our AI Assistant</h2>', unsafe_allow_html=True)

# Check for Groq API Key in secrets
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
    api_key_present = True
except (KeyError, FileNotFoundError):
    st.warning("`GROQ_API_KEY` not found in Streamlit secrets. The Q&A bot is disabled. Please add it to your secrets file.", icon="⚠️")
    api_key_present = False

# Placeholder for your public Google Sheet URL
# IMPORTANT: Make sure your sheet is public ("Anyone with the link can view")
google_sheet_url = "https://docs.google.com/spreadsheets/d/1Qsyn7n39z_tDoCUyF3C3a-k2jTym-2a--g6_7V2-S9U/edit#gid=0"


if api_key_present:
    # Load data and prepare context
    qa_data = load_data(google_sheet_url)
    if qa_data is not None:
        context = "\n".join([f"Q: {row['Question']}\nA: {row['Answer']}" for index, row in qa_data.iterrows()])
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages from history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # React to user input
        if prompt := st.chat_input("Ask about the program, fees, schedule..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)

            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                try:
                    # Initialize LangChain components
                    llm = ChatGroq(
                        model_name="llama-3.3-70b-versatile",
                        groq_api_key=groq_api_key
                    )
                    
                    system_prompt = """You are a helpful assistant for the Pragyan AI Executive Program. Answer the user's question based ONLY on the following information. If the answer is not in the information, say 'I do not have that information, please contact us for more details.'

                    <CONTEXT>
                    {context}
                    </CONTEXT>
                    """
                    
                    prompt_template = ChatPromptTemplate.from_messages([
                        ("system", system_prompt),
                        ("human", "{user_question}")
                    ])

                    output_parser = StrOutputParser()
                    
                    chain = prompt_template | llm | output_parser
                    
                    response = chain.invoke({
                        "context": context,
                        "user_question": prompt
                    })
                    
                    st.markdown(response)
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"An error occurred with the AI assistant: {e}")
    else:
        st.info("Q&A data could not be loaded. Please check the Google Sheet URL and permissions.")


st.markdown("<br><br>", unsafe_allow_html=True)


# --- Call to Action Section ---
st.markdown('<div class="highlight-card" style="background-color: #334155;">', unsafe_allow_html=True)
st.markdown('<h2 class="section-header">Secure Your Future in the Age of AI</h2>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center;">Enrollment is now open. Take the next step in your career.</p>', unsafe_allow_html=True)

cta_cols = st.columns([1, 1])

with cta_cols[0]:
    # Link button to a Google Drive document
    st.link_button(
        "📄 View Program Brochure",
        "https://drive.google.com/file/d/1JXtgnsfceX7doT8-mECEGjE_MXQgp9lK/view?usp=sharing",
        use_container_width=True,
    )
    
with cta_cols[1]:
    # Link button to redirect to Google Form
    st.link_button(
        "📝 Express Interest (Google Form)", 
        "https://forms.gle/YLKzVeEPsy685KvJA",
        use_container_width=True
    )
st.markdown('</div>', unsafe_allow_html=True)


# --- Footer ---
st.markdown(
    """
    <div class="footer">
        <p>© 2025 Pragyan AI. All rights reserved.</p>
        <p>
            <strong>Contact:</strong> 
            <a href="mailto:pragyan.ai.school@gmail.com">pragyan.ai.school@gmail.com</a> | 
            +91-9741007422 | 
            <a href="https://www.linkedin.com/in/sateesh-ambesange-3020185/" target="_blank">LinkedIn</a>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
