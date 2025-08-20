import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from PyPDF2 import PdfReader
import os
import asyncio
import nest_asyncio
nest_asyncio.apply()
try:
    from langchain_openai import ChatOpenAI  # type: ignore
except Exception:
    ChatOpenAI = None  # type: ignore
try:
    from langchain_anthropic import ChatAnthropic  # type: ignore
except Exception:
    ChatAnthropic = None  # type: ignore
try:
    from langchain_cohere import ChatCohere  # type: ignore
except Exception:
    ChatCohere = None  # type: ignore
# Ensure an asyncio event loop exists in Streamlit runner thread (Windows fix)
if os.name == "nt":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore[attr-defined]
    except Exception:
        pass
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Set the Google API Key in the environment
st.set_page_config(page_title="PDF QA App", page_icon="📄", layout="wide")

st.markdown(
    """
    <style>
    body {
        background-color: #eef2f3;
        color: #2c3e50;
        font-family: Arial, sans-serif;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        font-size: 16px;
        padding: 8px 16px;
        border: none;
        border-radius: 6px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #2980b9;
    }
    .stFileUploader {
        margin-bottom: 20px;
    }
    .stTextArea {
        margin-top: 10px;
    }

    /* Move sidebar to the right */
    :root {
        --sidebar-width: 22rem;
    }
    [data-testid="stSidebar"] {
        position: fixed;
        top: 0;
        right: 0;
        left: auto;
        height: 100%;
        min-width: var(--sidebar-width);
        max-width: var(--sidebar-width);
        border-left: 1px solid rgba(49, 51, 63, 0.2);
        border-right: none;
        background-color: inherit;
        z-index: 100;
    }
    /* Ensure main content leaves space on the right for the sidebar */
    [data-testid="stAppViewContainer"] .main {
        margin-left: 0 !important;
        margin-right: calc(var(--sidebar-width) + 1rem) !important;
    }
    /* Left-align the main block content rather than center */
    [data-testid="stAppViewContainer"] .main .block-container {
        max-width: 1200px;
        margin-left: 1rem !important;
        margin-right: calc(var(--sidebar-width) + 2rem) !important;
    }
    /* Keep header width aligned */
    [data-testid="stHeader"] {
        right: var(--sidebar-width);
        left: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# App Header
st.title("📄 PDF QA App with Generative AI")

# Main Q&A area centered below the title
left_pad, center_area, right_pad = st.columns([1, 2, 1])
with center_area:
    st.subheader("Ask a Question")
    st.caption("Tip: You can also click a suggested question below.")
    st.text_area(
        "Enter your question:",
        placeholder="E.g., What is the main topic of the document?",
        key="main_question",
    )

# File Upload Section
st.sidebar.header("📂 Upload a PDF")
uploaded_file = st.sidebar.file_uploader(
    "Choose a PDF file to extract content:",
    type=["pdf"],
)

# User Question Section
st.sidebar.header("❓ Controls")

# LLM provider and model selection
provider = st.sidebar.selectbox(
    "LLM Provider",
    ["Google Gemini", "OpenAI", "Anthropic", "Cohere"],
    index=0,
)
provider_to_models = {
    "Google Gemini": ["gemini-1.5-flash", "gemini-1.5-pro"],
    "OpenAI": ["gpt-4o", "gpt-4o-mini", "gpt-4.1"],
    "Anthropic": ["claude-3-5-sonnet-20240620", "claude-3-haiku-20240307"],
    "Cohere": ["command-r-plus", "command-r"]
}
model_name = st.sidebar.selectbox("Model", provider_to_models.get(provider, ["gemini-1.5-flash"]))

provider_api_key = st.sidebar.text_input(
    f"Enter your {provider} API key:", type="password", key="provider_api_key"
)

# Optionally set environment variable for selected provider
if provider_api_key:
    if provider == "Google Gemini":
        os.environ["GOOGLE_API_KEY"] = provider_api_key
    elif provider == "OpenAI":
        os.environ["OPENAI_API_KEY"] = provider_api_key
    elif provider == "Anthropic":
        os.environ["ANTHROPIC_API_KEY"] = provider_api_key
    elif provider == "Cohere":
        os.environ["COHERE_API_KEY"] = provider_api_key

# Removed question input from sidebar; question is entered in the right panel under the title
top_k = st.sidebar.slider("Top-k passages", min_value=2, max_value=10, value=4, step=1)
show_context = st.sidebar.checkbox("Show retrieved context", value=False)

@st.cache_data
def extract_text_from_pdf(pdf_file):
    """
    Extracts text from a PDF file using PyPDF2.
    """
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

@st.cache_data
def get_text_chunks(text):
    """
    Splits the loaded text into chunks for embedding and retrieval.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, chunk_overlap=1000
    )
    return text_splitter.split_text(text)

@st.cache_resource
def get_vector_store(text_chunks):
    '''
    Embeds the text chunks into a vector store for similarity search.
    '''
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    metadatas = [{"source": f"chunk_{i+1}"} for i in range(len(text_chunks))]
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings, metadatas=metadatas)
    return vector_store


def get_conversational_chain(model_name: str = "gemini-1.5-flash"):
    '''
    Creates a conversational chain for QA using LangChain and Google Generative AI.
    '''
    prompt_template = """
    Answer the question as detailed as possible from the provided context. Make sure to provide all the details.
    If the answer is not in the provided context, just say, "Answer is not available in the context." Don't provide a wrong answer.

    Context:\n{context}\n
    Question:\n{question}\n

    Answer:
    """

    if provider == "Google Gemini":
        model = ChatGoogleGenerativeAI(model=model_name, temperature=0.3)
    elif provider == "OpenAI" and ChatOpenAI is not None:
        model = ChatOpenAI(model=model_name, temperature=0.3)
    elif provider == "Anthropic" and ChatAnthropic is not None:
        model = ChatAnthropic(model=model_name, temperature=0.3)
    elif provider == "Cohere" and ChatCohere is not None:
        model = ChatCohere(model=model_name, temperature=0.3)
    else:
        model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)


def suggest_questions(text_chunks, num_questions: int = 5):
    """
    Generate suggested questions based on document content.
    """
    try:
        model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
        selected_chunks = text_chunks[:5] if len(text_chunks) > 5 else text_chunks
        context = "\n\n".join(selected_chunks)
        prompt = (
            f"You are a helpful assistant. Based on the given context, propose {num_questions} specific, high-value questions "
            "a user might ask about this document. Keep each question under 18 words.\n"
            "Return one question per line, with no numbering or extra text.\n\nContext:\n" + context
        )
        resp = model.invoke(prompt)
        text = getattr(resp, "content", str(resp))
        lines = [q.strip("- •\t ") for q in text.splitlines() if q.strip()]
        unique = []
        for q in lines:
            if q not in unique:
                unique.append(q)
        return unique[:num_questions] if unique else []
    except Exception:
        return []


def summarize_document(text_chunks, preset: str = "Executive"):
    """
    Creates a concise summary from the first few chunks with style presets.
    """
    try:
        model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.2)
        selected_chunks = text_chunks[:5] if len(text_chunks) > 5 else text_chunks
        context = "\n\n".join(selected_chunks)
        if preset == "Executive":
            style = (
                "Create an executive summary (120-200 words) in 5-8 concise bullet points aimed at business leaders. "
                "Emphasize objectives, risks, deadlines, decisions, and recommendations. Avoid technical jargon."
            )
        elif preset == "Technical":
            style = (
                "Create a technical summary in 8-12 bullet points. Focus on mechanisms, algorithms, parameters, constraints, "
                "figures, edge cases, and step-by-step procedures. Include concrete numbers when present."
            )
        else:  # Student
            style = (
                "Create a student-friendly summary in 5-7 bullet points using simple language. "
                "Include definitions, key takeaways, and brief examples where helpful."
            )
        summary_prompt = style + "\n\nContext:\n" + context
        resp = model.invoke(summary_prompt)
        return getattr(resp, "content", str(resp))
    except Exception as e:
        return f"Summary failed: {e}"


# Build index when PDF and provider API key are provided
vector_store = None
text_chunks = None
if uploaded_file and provider_api_key:
    with st.spinner("Indexing your PDF..."):
        try:
            raw_text = extract_text_from_pdf(uploaded_file)
            text_chunks = get_text_chunks(raw_text)
            vector_store = get_vector_store(text_chunks)
        except Exception as e:
            st.error(f"Failed to process PDF: {e}")

# Suggested questions and summary feature (rendered in sidebar on the right)
if vector_store is not None:
    st.sidebar.subheader("🔎 Quick questions")
    dynamic_suggestions = suggest_questions(text_chunks, num_questions=5)
    suggestions = dynamic_suggestions or [
        "What is this document about?",
        "List important dates and deadlines.",
        "Who are the key stakeholders?",
        "What are the eligibility criteria?",
        "Summarize the main steps.",
    ]
    for i, s in enumerate(suggestions):
        if st.sidebar.button(s, key=f"sugg_{i}"):
            st.session_state["user_question"] = s

    with st.sidebar.expander("✨ Document summary", expanded=False):
        preset = st.selectbox("Summary style", ["Executive", "Technical", "Student"], index=0, key="summary_style")
        if st.button("Generate summary", key="gen_summary_btn"):
            summary_text = summarize_document(text_chunks, preset=preset)
            st.write(summary_text)
            st.download_button("Download summary", data=summary_text, file_name="summary.txt", key="dl_summary")

# Question answering flow
# Take question from the main right panel if present; fallback to suggested question state
effective_question = (st.session_state.get("main_question") or st.session_state.get("user_question") or "").strip()
if vector_store is not None and effective_question:
    with st.spinner("Answering your question..."):
        try:
            results = vector_store.similarity_search_with_score(effective_question, k=top_k)
            docs = [doc for doc, _ in results]
            chain = get_conversational_chain(model_name=model_name)
            response = chain({"input_documents": docs, "question": effective_question}, return_only_outputs=True)

            # Display the results in the centered panel
            with center_area:
                st.success("Answer Generated!")
                st.subheader("Your Question:")
                st.write(effective_question)

                answer_text = response.get("output_text", "")
                st.subheader("Generated Answer:")
                for line in answer_text.split("\n"):
                    if line.strip():
                        st.write(line)
                if answer_text:
                    st.download_button("Download answer", data=answer_text, file_name="answer.md", key="dl_answer")

            # Optional: Show sources and context with scores in sidebar
            with st.sidebar.expander("Sources"):
                for d in docs:
                    st.write(d.metadata.get("source", "Unknown"))

            if show_context:
                with st.sidebar.expander("Retrieved context"):
                    for idx, (d, score) in enumerate(results, start=1):
                        st.markdown(f"**Chunk {idx}** — score: {score:.4f} — source: {d.metadata.get('source', 'Unknown')}")
                        preview = d.page_content[:600] + ("..." if len(d.page_content) > 600 else "")
                        st.write(preview)

        except Exception as e:
            st.error(f"An error occurred: {e}")
elif not provider_api_key or not uploaded_file:
    if not provider_api_key:
        st.sidebar.warning("Please provide your API key in the sidebar.")
    if not uploaded_file:
        st.sidebar.warning("Please upload a PDF file.")

# Footer
st.markdown(
    """
    ---
    🌟 Powered by LangChain and Google Generative AI
    """
)
