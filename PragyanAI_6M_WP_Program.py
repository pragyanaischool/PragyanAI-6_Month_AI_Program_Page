import streamlit as st
import pandas as pd
import requests
import io
import logging
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

# --- Page Configuration ---
st.set_page_config(
    page_title="PragyanAI - 6 Month Executive Program in Generative & Agentic AI",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- SESSION STATE INITIALIZATION ---
# Initialize session state variables at the top to ensure they are always available.
if "messages" not in st.session_state:
    st.session_state.messages = []
if 'user_name' not in st.session_state:
    st.session_state.user_name = ""
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'chat_dialog_open' not in st.session_state:
    st.session_state.chat_dialog_open = False


# --- Custom CSS for Styling ---
st.markdown("""
<style>
    /* General Styles */
    .stApp {
        background-color: #0f172a;
        color: #e2e8f0; /* Lighter text color for better readability */
    }
    h1, h2, h3 {
        color: #ffffff;
        font-weight: 700;
    }
    p, .stMarkdown {
        color: #e2e8f0;
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
    
    /* Chat Message Styles */
    .stChatMessage {
        background-color: rgba(30, 41, 59, 0.5);
        border-radius: 0.75rem;
        padding: 1rem;
        border: 1px solid #334155;
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


# --- Data Loading and Caching Functions ---
@st.cache_data(ttl=3600)
def load_faq_data(sheet_url):
    """Loads Q&A data from a public Google Sheet."""
    try:
        csv_url = sheet_url.replace("/edit?usp=sharing", "/export?format=csv")
        df = pd.read_csv(csv_url)
        if df.shape[1] < 2:
            st.error("The Google Sheet needs at least two columns for questions and answers.")
            return None
        df = df.iloc[:, [0, 1]]
        df.columns = ['FAQs', 'Answere']
        df.dropna(inplace=True)
        return df
    except Exception as e:
        st.error(f"Failed to load data from Google Sheet: {e}", icon="📄")
        return None

@st.cache_data(ttl=3600)
def load_pdf_from_gdrive(drive_url):
    """Downloads and extracts text from a PDF in Google Drive."""
    try:
        file_id = drive_url.split('/d/')[1].split('/')[0]
        download_url = f'https://drive.google.com/uc?export=download&id={file_id}'
        response = requests.get(download_url)
        response.raise_for_status()
        
        pdf_file = io.BytesIO(response.content)
        reader = PdfReader(pdf_file)
        text = "".join(page.extract_text() for page in reader.pages)
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_text(text)
        
        return [Document(page_content=split, metadata={"source": "program_brochure"}) for split in splits]
    except Exception as e:
        st.error(f"Failed to load PDF from Google Drive: {e}", icon="📄")
        return None

@st.cache_resource
def get_embeddings_model():
    """Initializes and returns the sentence transformer embeddings model."""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource
def create_vector_store(_faq_df, _pdf_docs, _embeddings):
    """Creates a FAISS vector store from the FAQ DataFrame and PDF documents."""
    documents = []
    if _faq_df is not None:
        documents.extend([
            Document(page_content=f"Question: {row['FAQs']}\nAnswer: {row['Answere']}", metadata={"source": "faq_sheet"})
            for _, row in _faq_df.iterrows()
        ])
    if _pdf_docs is not None:
        documents.extend(_pdf_docs)
    
    return FAISS.from_documents(documents, _embeddings) if documents else None

# --- RAG Chain Helper Functions ---
def get_history(_):
    return "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.get('messages', [])])

def get_name(_):
    return st.session_state.get('user_name', "")

# --- Pre-load Data on App Start ---
def initialize_knowledge_base():
    if st.session_state.vector_store is None:
        with st.spinner("Preparing the AI Advisor, please wait..."):
            faq_sheet_url = "https://docs.google.com/spreadsheets/d/14NTraereEwWwLyhycjCP0TKJ2-a6eY38xjy5EbAN-jM/edit?usp=sharing"
            brochure_drive_url = "https://drive.google.com/file/d/1JXtgnsfceX7doT8-mECEGjE_MXQgp9lK/view?usp=sharing"

            faq_data = load_faq_data(faq_sheet_url)
            pdf_docs = load_pdf_from_gdrive(brochure_drive_url)
            
            if faq_data is not None or pdf_docs is not None:
                embeddings = get_embeddings_model()
                st.session_state.vector_store = create_vector_store(faq_data, pdf_docs, embeddings)

initialize_knowledge_base()

# --- CHAT DIALOG FUNCTION ---
@st.dialog("PragyanAI - AI Marketing Advisor")
def run_chat_dialog():
    """Contains the entire chat UI and logic within a Streamlit Dialog."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about career impact, curriculum, fees..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                retriever = st.session_state.vector_store.as_retriever(search_kwargs={'k': 5})
                groq_api_key = st.secrets["GROQ_API_KEY"]
                llm = ChatGroq(model_name="llama-3.3-70b-versatile", groq_api_key=groq_api_key)
                
                system_prompt_template = """You are an expert marketing advisor and enthusiastic career coach for the Pragyan AI Executive Program. Your name is PragyanAI Bot. Your goal is to advise a user named {user_name} on why this program is a critical step for their career.
                Follow these rules strictly:
                1. Persona: Be persuasive, encouraging, and an expert on AI career paths. Frame every answer to highlight the *value and benefits* for a working professional. Explain *why* a feature is important for their career growth.
                2. Acknowledge and Address: Always address the user by their name, {user_name}.
                3. Data Source: Base your answers ONLY on the retrieved context from the program brochure and FAQ sheet.
                4. Marketing Spin: Don't just state facts. Explain the 'so what'. For example, if asked about MLOps, explain that mastering it moves them from just building models to deploying scalable, production-ready AI, a skill that commands a premium salary.
                5. Context is Key: Use the conversation history to maintain a natural, flowing dialogue.
                6. Handle Unknowns: If the context doesn't have the answer, you MUST say: "That's an excellent and very specific question, {user_name}. While I don't have the details on that, it's something our program director can certainly clarify. I highly recommend reaching out to them at pragyan.ai.school@gmail.com for a direct answer." Do not invent information.
                <CONVERSATION_HISTORY>{chat_history}</CONVERSATION_HISTORY>
                <RETRIEVED_CONTEXT>{context}</RETRIEVED_CONTEXT>
                Given the context and conversation history, answer the user's question: {user_question}"""
                
                prompt_template = ChatPromptTemplate.from_template(system_prompt_template)
                
                rag_chain = (
                    RunnableParallel({
                        "context": retriever | (lambda docs: "\n\n".join(doc.page_content for doc in docs)),
                        "user_question": RunnablePassthrough(),
                        "chat_history": RunnableLambda(get_history),
                        "user_name": RunnableLambda(get_name)
                    })
                    | prompt_template
                    | llm
                    | StrOutputParser()
                )
                
                response = rag_chain.invoke(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun() # Rerun to update the chat history display in the dialog
            except Exception as e:
                # --- THIS IS THE FIX ---
                # Log the detailed, sensitive error to the console for debugging
                logging.error(f"AI Assistant Error: {e}")
                # Show a generic, safe message to the user
                st.error("I'm sorry, I encountered a connection issue. Please check your API key or try again later.")


# --- UI LAYOUT ---

# Header
col1, col2, col3 = st.columns([2, 5, 2])
with col2:
    st.image("PragyanAI_Transperent_github.png", use_container_width=True)

# Hero Section
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<h1 style="text-align: center;">Executive Program in <span class="gradient-text">Generative & Agentic AI</span></h1>', unsafe_allow_html=True)
st.markdown('<h3 style="text-align: center; color: #94a3b8;">A 6-Month Weekend Accelerator for Senior IT Professionals (4+ Years Experience)</h3>', unsafe_allow_html=True)
st.markdown("""<p style="text-align: center; max-width: 700px; margin: auto; padding-top: 1rem;">
    In a tech landscape where AI is actively reshaping job roles, proactive upskilling is essential. This program is a decisive move to future-proof your career and transition from an IT professional to an <strong>AI Solution Architect and Leader.</strong>
</p>""", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# Program Highlights
st.markdown('<h2 class="section-header">Program Highlights</h2>', unsafe_allow_html=True)
cols = st.columns(4)
highlights = {"Duration": "6 Months", "Schedule": "Sat & Sun (Live)", "Audience": "IT Professionals (4+ Yrs)", "Fee": "₹1,00,000"}
for i, (title, value) in enumerate(highlights.items()):
    with cols[i]:
        st.markdown(f'<div class="highlight-card"><h3>{title}</h3><p style="font-size: 1.5rem; color: #f97316; font-weight: 600;">{value}</p></div>', unsafe_allow_html=True)
st.markdown("<br><br>", unsafe_allow_html=True)

# Curriculum Section
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
st.markdown('<h2 class="section-header">Have Questions? Ask PragyanAI AI Marketing Advisor</h2>', unsafe_allow_html=True)

# Check for Groq API Key in secrets
try:
    #st.secrets["GROQ_API_KEY"]
    api_key_present = True
except (KeyError, FileNotFoundError):
    api_key_present = False

if not api_key_present:
    st.warning(" LLM API KEY not found in Streamlit secrets. The Q&A bot is disabled.", icon="⚠️")
elif st.session_state.vector_store is None:
    st.warning("The knowledge base for the AI Advisor could not be loaded. Please check data sources.", icon="🧠")
else:
    st.session_state.user_name = st.text_input("Please enter your name to start the chat:", st.session_state.user_name)
    if st.session_state.user_name:
        if st.button("🚀 Launch PRAGYANAI -  Program Advisor"):
            st.session_state.chat_dialog_open = True
    
    if st.session_state.chat_dialog_open:
        run_chat_dialog()


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
st.markdown("""<div class="footer">
    <p>© 2025 Pragyan AI. All rights reserved.</p>
    <p>
        <strong>Contact:</strong> 
        <a href="mailto:pragyan.ai.school@gmail.com">pragyan.ai.school@gmail.com</a> | 
        +91-9741007422 | 
        <a href="https://www.linkedin.com/in/sateesh-ambesange-3020185/" target="_blank">LinkedIn</a>
    </p>
</div>""", unsafe_allow_html=True)
