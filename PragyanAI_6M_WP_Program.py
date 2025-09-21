import streamlit as st
import pandas as pd
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- Page Configuration ---
st.set_page_config(
    page_title="PragyanAI - 6 Month Executive Program in Generative & Agentic AI",
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
    
    /* Expander Styles */
    .st-emotion-cache-1h9usn1 p {
        font-size: 1.1rem;
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
        csv_url = sheet_url.replace("/edit?usp=sharing", "/export?format=csv")
        df = pd.read_csv(csv_url)
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
curriculum_details = {
    "1. Advanced Python & Data Engineering": ["Advanced Data Wrangling (Pandas, Polars, Dask)", "Multi-Modal Data Processing (Images, Audio, Video)", "Web Automation & Data Extraction (Selenium, BeautifulSoup)"],
    "2. Applied Data Science & Analytics": ["Practical Statistics for ML (Hypothesis Testing, ANOVA)", "Advanced Interactive Data Visualization (Plotly, Seaborn)", "Automated EDA & Feature Engineering"],
    "3. BI & Big Data Architecture": ["Business Intelligence (BI) Fundamentals", "Modern Data Architectures (Data Lakes, Warehouses)", "Hands-on with Open-Source BI Platforms (Metabase, Superset)"],
    "4. Practical Machine Learning & MLOps": ["Advanced Algorithms (XGBoost, LightGBM, CatBoost)", "Model Optimization & Hyperparameter Tuning (Optuna)", "MLOps & Deployment (Streamlit, Gradio, FastAPI)", "Explainable AI (XAI) with SHAP & LIME"],
    "5. Applied Deep Learning & Computer Vision": ["Core Architectures: CNNs, RNNs, Transformers", "Advanced CV Tasks: Object Detection, Segmentation, OCR", "Working with SOTA Models (YOLO, SAM, ViT)", "Edge AI & Model Optimization (TensorFlow Lite)"],
    "6. NLP & Conversational AI": ["Modern Text Representation (BERT & Embeddings)", "Core NLP Applications (NER, Sentiment Analysis)", "Building Conversational AI & Chatbots"],
    "7. Generative AI & Large Language Models": ["Deep Dive into the Transformer Architecture", "Retrieval Augmented Generation (RAG)", "LLM Fine-Tuning with PEFT & LoRA", "Advanced Prompt Engineering (Chain-of-Thought, ReAct)"],
    "8. Agentic AI & MVP Development": ["AI Agent Architecture & Frameworks (CrewAI, AutoGen)", "Multi-Agent Collaborative Systems", "Rapid AI-Powered MVP Prototyping", "From Developer to AI Solution Designer"]
}
for domain, sub_topics in curriculum_details.items():
    with st.expander(f"**{domain}**"):
        for topic in sub_topics:
            st.markdown(f"- {topic}")
st.markdown("<br><br>", unsafe_allow_html=True)

# --- Q&A Section ---
st.markdown('<h2 class="section-header">Have Questions? Ask our AI Assistant</h2>', unsafe_allow_html=True)

# Check for Groq API Key in secrets
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
    api_key_present = True
except (KeyError, FileNotFoundError):
    api_key_present = False

google_sheet_url = "https://docs.google.com/spreadsheets/d/14NTraereEwWwLyhycjCP0TKJ2-a6eY38xjy5EbAN-jM/edit?usp=sharing"

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if 'user_name' not in st.session_state:
    st.session_state.user_name = ""

# Get user's name
st.session_state.user_name = st.text_input("Please enter your name to start the chat:", st.session_state.user_name)

# Display prior chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if api_key_present and st.session_state.user_name:
    qa_data = load_data(google_sheet_url)
    if qa_data is not None:
        context = "\n".join([f"Q: {row['Question']}\nA: {row['Answer']}" for index, row in qa_data.iterrows()])
        
        if prompt := st.chat_input("Ask about the program, fees, schedule..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                try:
                    chat_history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages])

                    llm = ChatGroq(model_name="llama-3.3-70b-versatile", groq_api_key=groq_api_key)
                    
                    system_prompt = """You are a friendly and helpful assistant for the Pragyan AI Executive Program. Your name is PragyanAI Bot. Your goal is to answer questions for a user named {user_name}.

                    Follow these rules strictly:
                    1. Be conversational and encouraging. Address the user by their name, {user_name}, when appropriate.
                    2. Base your answers ONLY on the provided context from the FAQ sheet.
                    3. Consider the ongoing conversation history to maintain context and avoid repetition.
                    4. If the answer is not in the provided context, you MUST say: "That's a great question, {user_name}! I don't have that specific information right now, but please reach out to our team at pragyan.ai.school@gmail.com for more details." Do not make up answers.

                    <CONVERSATION_HISTORY>
                    {chat_history}
                    </CONVERSATION_HISTORY>

                    <FAQ_CONTEXT>
                    {context}
                    </FAQ_CONTEXT>
                    """
                    
                    prompt_template = ChatPromptTemplate.from_messages([
                        ("system", system_prompt),
                        ("human", "{user_question}")
                    ])

                    output_parser = StrOutputParser()
                    chain = prompt_template | llm | output_parser
                    
                    response = chain.invoke({
                        "context": context,
                        "user_question": prompt,
                        "user_name": st.session_state.user_name,
                        "chat_history": chat_history_str
                    })
                    
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"An error occurred with the AI assistant: {e}")
    else:
        st.info("Q&A data could not be loaded. Please check the Google Sheet URL and permissions.")
elif not api_key_present:
    st.warning("`GROQ_API_KEY` not found in Streamlit secrets. The Q&A bot is disabled. Please add it to your secrets file.", icon="⚠️")
    st.chat_input("Q&A Bot is disabled. API key is missing.", disabled=True)
else: # API key is present but name is not
    st.info("Please enter your name above to activate the AI Assistant.")
    st.chat_input("Enter your name to begin chat.", disabled=True)


st.markdown("<br><br>", unsafe_allow_html=True)

# --- Call to Action Section ---
st.markdown('<div class="highlight-card" style="background-color: #334155;">', unsafe_allow_html=True)
st.markdown('<h2 class="section-header">Secure Your Future in the Age of AI</h2>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center;">Enrollment is now open. Take the next step in your career.</p>', unsafe_allow_html=True)
cta_cols = st.columns([1, 1])
with cta_cols[0]:
    st.link_button("📄 View Program Brochure", "https://drive.google.com/file/d/1JXtgnsfceX7doT8-mECEGjE_MXQgp9lK/view?usp=sharing", use_container_width=True)
with cta_cols[1]:
    st.link_button("📝 Express Interest (Google Form)", "https://forms.gle/YLKzVeEPsy685KvJA", use_container_width=True)
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
