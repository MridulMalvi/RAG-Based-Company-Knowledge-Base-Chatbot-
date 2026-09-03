"""
app.py — Enterprise-Grade Streamlit UI for Nexora Technologies Knowledge Base Chatbot
=====================================================================================
A clean, Tailwind-inspired corporate interface for querying company documentation,
uploading documents (PDF, TXT, DOCX, MD), and inspecting citations with zero hallucinations.
"""

import streamlit as st
import time
import traceback
from pathlib import Path

# Import RAG pipeline components
import rag_pipeline as rag

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Nexora Technologies — Knowledge Base",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Enterprise Tailwind-Inspired Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* ── Typography & Base Styles ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        color: #F8FAFC;
    }

    /* Target content elements specifically without breaking Streamlit icon fonts */
    p, div.stMarkdown, h1, h2, h3, h4, h5, h6, label {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    /* ── Protect Streamlit Native Icon Webfonts from text-font overrides ── */
    [data-testid*="Icon"],
    [data-testid*="icon"],
    [data-testid="stExpanderToggleIcon"],
    .material-symbols-rounded,
    .material-icons,
    [class*="material-symbols"],
    [class*="material-icons"] {
        font-family: 'Material Symbols Rounded', 'Material Icons', sans-serif !important;
        speak: never;
        font-style: normal;
        font-weight: normal;
        line-height: 1;
        letter-spacing: normal;
        text-transform: none;
        display: inline-block;
        white-space: nowrap;
        word-wrap: normal;
        direction: ltr;
        -webkit-font-smoothing: antialiased;
    }

    /* ── Enterprise Dark Background ── */
    .stApp {
        background-color: #0B0F19 !important;
    }

    /* ── Header Bar ── */
    header[data-testid="stHeader"] {
        background-color: #111827 !important;
        border-bottom: 1px solid #1F2937 !important;
    }

    /* ── Sidebar Styling ── */
    [data-testid="stSidebar"] {
        background-color: #111827 !important;
        border-right: 1px solid #1F2937 !important;
        padding-top: 1rem !important;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4 {
        color: #F8FAFC !important;
        font-weight: 600 !important;
        letter-spacing: -0.01em !important;
    }

    [data-testid="stSidebar"] hr {
        margin: 1.2rem 0 !important;
        border-color: #1F2937 !important;
    }

    /* ── Main Container & Layout ── */
    .block-container {
        padding-top: 1.8rem !important;
        padding-bottom: 3.5rem !important;
        max-width: 1100px !important;
    }

    /* ── Header Section ── */
    .enterprise-header {
        background-color: #111827;
        border: 1px solid #1F2937;
        border-radius: 8px;
        padding: 20px 24px;
        margin-bottom: 24px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
    }
    .badge-corporate {
        display: inline-flex;
        align-items: center;
        background-color: #1E3A8A;
        color: #93C5FD;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 3px 9px;
        border-radius: 4px;
        border: 1px solid #2563EB;
        margin-bottom: 8px;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }
    .main-title {
        font-size: 1.65rem;
        font-weight: 700;
        color: #F8FAFC;
        margin: 0 0 6px 0;
        letter-spacing: -0.02em;
    }
    .main-subtitle {
        font-size: 0.92rem;
        color: #94A3B8;
        margin: 0;
        line-height: 1.5;
    }

    /* ── Chat Messages ── */
    .chat-card-user {
        background-color: #2563EB;
        color: #FFFFFF !important;
        padding: 13px 18px;
        border-radius: 8px 8px 2px 8px;
        margin: 14px 0 10px auto;
        max-width: 80%;
        font-size: 0.94rem;
        line-height: 1.55;
        box-shadow: 0 2px 6px rgba(37, 99, 235, 0.3);
    }
    .chat-card-user strong {
        color: #DBEAFE;
    }

    .chat-card-bot {
        background-color: #1E293B;
        color: #F8FAFC !important;
        border: 1px solid #334155;
        padding: 16px 20px;
        border-radius: 8px 8px 8px 2px;
        margin: 10px auto 14px 0;
        max-width: 95%;
        font-size: 0.94rem;
        line-height: 1.6;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.25);
    }

    /* ── Fallback Sentinel Alert ── */
    .alert-no-info {
        background-color: #451A03;
        border: 1px solid #B45309;
        color: #FDE68A;
        padding: 14px 18px;
        border-radius: 6px;
        font-size: 0.9rem;
        font-weight: 500;
        margin: 10px 0;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* ── Citations Grid & Cards ── */
    .source-card {
        background-color: #0F172A;
        border: 1px solid #334155;
        border-radius: 6px;
        padding: 12px 14px;
        margin-bottom: 10px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2);
        height: 100%;
        transition: border-color 0.15s ease;
    }
    .source-card:hover {
        border-color: #60A5FA;
    }
    .source-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 6px;
    }
    .source-tag {
        font-size: 0.75rem;
        font-weight: 600;
        color: #93C5FD;
        background-color: #1E3A8A;
        padding: 2px 7px;
        border-radius: 4px;
        border: 1px solid #2563EB;
    }
    .chunk-tag {
        font-size: 0.72rem;
        font-weight: 500;
        color: #CBD5E1;
        background-color: #334155;
        padding: 2px 6px;
        border-radius: 4px;
    }
    .source-title {
        font-weight: 600;
        font-size: 0.85rem;
        color: #F1F5F9;
        margin-bottom: 4px;
    }
    .source-preview {
        font-size: 0.79rem;
        color: #94A3B8;
        line-height: 1.45;
        display: -webkit-box;
        -webkit-line-clamp: 3;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }

    /* ── Targeted Button Styling via Streamlit Component Keys ── */
    /* Primary action buttons */
    div.st-key-btn_process_uploads button {
        background-color: #2563EB !important;
        color: #FFFFFF !important;
        border: 1px solid #3B82F6 !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        padding: 7px 14px !important;
        box-shadow: 0 1px 3px rgba(37, 99, 235, 0.3) !important;
        transition: all 0.15s ease !important;
    }
    div.st-key-btn_process_uploads button:hover {
        background-color: #1D4ED8 !important;
        border-color: #60A5FA !important;
    }

    /* Secondary utility buttons */
    div.st-key-btn_rebuild button,
    div.st-key-btn_clear button {
        background-color: #1E293B !important;
        color: #F1F5F9 !important;
        border: 1px solid #334155 !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
        font-size: 0.82rem !important;
        padding: 6px 12px !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.2) !important;
        transition: all 0.15s ease !important;
    }
    div.st-key-btn_rebuild button:hover,
    div.st-key-btn_clear button:hover {
        background-color: #334155 !important;
        color: #FFFFFF !important;
        border-color: #475569 !important;
    }

    /* Sidebar example question buttons */
    div[class*="st-key-ex_"] button {
        background-color: #1E293B !important;
        color: #CBD5E1 !important;
        border: 1px solid #334155 !important;
        border-radius: 6px !important;
        font-weight: 400 !important;
        font-size: 0.8rem !important;
        text-align: left !important;
        padding: 8px 10px !important;
        margin-bottom: 4px !important;
        transition: all 0.15s ease !important;
        display: block !important;
        white-space: normal !important;
        word-break: break-word !important;
    }
    div[class*="st-key-ex_"] button:hover {
        border-color: #3B82F6 !important;
        background-color: #1E3A8A !important;
        color: #93C5FD !important;
    }

    /* ── Chat Input Styling ── */
    [data-testid="stBottom"] {
        background-color: #0B0F19 !important;
    }
    [data-testid="stChatInput"] {
        background-color: transparent !important;
        border: none !important;
        padding: 8px 0 12px 0 !important;
    }
    /* Outer chat input box container */
    [data-testid="stChatInput"] > div {
        background-color: #1E293B !important;
        border: 1px solid #475569 !important;
        border-radius: 8px !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.3) !important;
        padding: 4px 8px !important;
        transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
    }
    [data-testid="stChatInput"] > div:focus-within {
        border-color: #3B82F6 !important;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.25) !important;
    }
    /* Text input inside the chat box */
    [data-testid="stChatInput"] textarea {
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
        color: #F8FAFC !important;
        font-size: 0.93rem !important;
        line-height: 1.5 !important;
        padding: 8px 10px !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: #94A3B8 !important;
        font-size: 0.88rem !important;
    }
    /* Send button inside the chat box */
    [data-testid="stChatInput"] button {
        background-color: transparent !important;
        color: #60A5FA !important;
        border: none !important;
        box-shadow: none !important;
        padding: 6px !important;
        align-self: center !important;
    }
    [data-testid="stChatInput"] button:hover {
        color: #93C5FD !important;
        background-color: rgba(59, 130, 246, 0.15) !important;
        border-radius: 6px !important;
    }

    /* ── Streamlit Expanders ── */
    [data-testid="stExpander"] {
        background-color: #111827 !important;
        border: 1px solid #334155 !important;
        border-radius: 6px !important;
        margin-top: 6px !important;
        margin-bottom: 6px !important;
        overflow: hidden !important;
    }
    [data-testid="stExpander"] summary {
        background-color: #111827 !important;
        color: #E2E8F0 !important;
        cursor: pointer !important;
    }
    [data-testid="stExpander"] summary:hover {
        color: #60A5FA !important;
    }
    [data-testid="stExpander"] summary p {
        color: #F1F5F9 !important;
        font-size: 0.84rem !important;
        font-weight: 500 !important;
        margin: 0 !important;
        padding-left: 6px !important;
    }
    [data-testid="stExpanderDetails"] {
        padding: 10px 12px !important;
        background-color: #0F172A !important;
        border-top: 1px solid #1F2937 !important;
    }

    /* ── File Uploader Styling ── */
    div.st-key-file_uploader {
        margin-top: 4px !important;
        margin-bottom: 8px !important;
    }
    div.st-key-file_uploader section[data-testid="stFileUploadDropzone"] {
        background-color: #111827 !important;
        border: 1px dashed #475569 !important;
        border-radius: 8px !important;
        padding: 14px 10px !important;
    }
    div.st-key-file_uploader section[data-testid="stFileUploadDropzone"]:hover {
        border-color: #3B82F6 !important;
        background-color: #1E293B !important;
    }
    div.st-key-file_uploader section[data-testid="stFileUploadDropzone"] button[data-testid="stBaseButton-secondary"],
    div.st-key-file_uploader section[data-testid="stFileUploadDropzone"] button {
        background-color: #1E293B !important;
        color: #F8FAFC !important;
        border: 1px solid #475569 !important;
        border-radius: 6px !important;
        font-size: 0.8rem !important;
        padding: 4px 12px !important;
    }
    div.st-key-file_uploader section[data-testid="stFileUploadDropzone"] button:hover {
        background-color: #334155 !important;
        border-color: #60A5FA !important;
    }
    div.st-key-file_uploader [data-testid="stUploadedFile"] {
        background-color: #1E293B !important;
        border: 1px solid #334155 !important;
        border-radius: 6px !important;
        padding: 6px 10px !important;
        margin-top: 6px !important;
    }
    div.st-key-file_uploader [data-testid="stUploadedFile"] span {
        color: #F8FAFC !important;
        font-size: 0.8rem !important;
    }

    /* ── Indexed Files List Styling ── */
    .indexed-files-list {
        display: flex;
        flex-direction: column;
        gap: 6px;
        padding: 2px 0;
    }
    .indexed-file-item {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 5px;
        padding: 6px 9px;
        font-size: 0.79rem;
    }
    .indexed-file-name {
        color: #F1F5F9;
        font-weight: 500;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        max-width: 175px;
    }
    .indexed-file-ext {
        color: #93C5FD;
        background-color: #1E3A8A;
        font-size: 0.68rem;
        font-weight: 600;
        padding: 1px 5px;
        border-radius: 3px;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }

    /* ── Metrics Cards ── */
    [data-testid="stMetric"] {
        background-color: #111827 !important;
        border: 1px solid #1F2937 !important;
        border-radius: 6px !important;
        padding: 8px 12px !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.74rem !important;
        color: #94A3B8 !important;
        font-weight: 500 !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.15rem !important;
        color: #F8FAFC !important;
        font-weight: 700 !important;
    }

    /* ── Code Blocks / Pre ── */
    pre, code {
        background-color: #0F172A !important;
        color: #E2E8F0 !important;
        border: 1px solid #1F2937 !important;
    }

    /* ── Clean Scrollbars ── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #111827; }
    ::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #475569; }

    /* Hide Streamlit footer */
    #MainMenu, footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "llm" not in st.session_state:
    st.session_state.llm = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "kb_status" not in st.session_state:
    st.session_state.kb_status = "not_loaded"
if "kb_error" not in st.session_state:
    st.session_state.kb_error = ""
if "doc_stats" not in st.session_state:
    st.session_state.doc_stats = {}
if "upload_success_msg" not in st.session_state:
    st.session_state.upload_success_msg = ""
if "uploaded_file_names" not in st.session_state:
    st.session_state.uploaded_file_names = []


# ---------------------------------------------------------------------------
# Pipeline Initialization (Cached for zero-delay across reruns)
# ---------------------------------------------------------------------------
@st.cache_resource(
    show_spinner="Initializing Enterprise Knowledge Base..."
)
def _init_pipeline():
    """
    Initializes embeddings, vectorstore, and LLM.
    Cached via @st.cache_resource so subsequent reruns are instant.
    """
    embeddings = rag.get_embeddings()
    if not rag.FAISS_INDEX_PATH.exists():
        docs = rag.load_documents()
        chunks = rag.split_documents(docs)
        vs = rag.build_vectorstore(chunks, embeddings)
    else:
        vs = rag.load_vectorstore(embeddings)
    llm = rag.get_llm()
    return embeddings, vs, llm


def is_no_info_response(answer: str) -> bool:
    """Check if the answer matches the fallback sentinel."""
    return rag.NO_INFO_RESPONSE.lower() in answer.lower()


# ---------------------------------------------------------------------------
# Sidebar: Professional Layout & Management Controls
# ---------------------------------------------------------------------------
with st.sidebar:
    # ── Brand Section ──
    st.markdown("### 🏢 Nexora Technologies")
    st.caption("Enterprise Knowledge Base & AI Assistant")
    st.markdown("---")

    # ── Knowledge Base Status ──
    st.markdown("#### System Status")
    if st.session_state.kb_status == "ready":
        st.success("Knowledge Base is Online", icon="✅")
        stats = st.session_state.doc_stats
        
        m1, m2 = st.columns(2)
        with m1:
            st.metric("Documents", stats.get("file_count", 0))
        with m2:
            st.metric("Top-K", rag.TOP_K_RESULTS)

        files = stats.get("files", [])
        with st.expander(f"Indexed Files ({len(files)})", expanded=False):
            if files:
                file_items = []
                for fname in files:
                    ext = fname.split(".")[-1].upper() if "." in fname else "DOC"
                    file_items.append(
                        f'<div class="indexed-file-item">'
                        f'<span class="indexed-file-name" title="{fname}">📄 {fname}</span>'
                        f'<span class="indexed-file-ext">{ext}</span>'
                        f'</div>'
                    )
                st.markdown(
                    f'<div class="indexed-files-list">{"".join(file_items)}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.caption("No files indexed.")
    elif st.session_state.kb_status == "loading":
        st.info("Loading knowledge base...", icon="⏳")
    elif st.session_state.kb_status == "error":
        st.error(f"Status: Offline\n{st.session_state.kb_error}", icon="❌")
    else:
        st.info("Status: Ready to connect", icon="ℹ️")

    st.markdown("---")

    # ── Document Ingestion ──
    st.markdown("#### Ingest Documents")
    kb_ready = st.session_state.kb_status == "ready"

    uploaded_files = st.file_uploader(
        "Upload files (.pdf, .txt, .docx, .md)",
        type=["pdf", "txt", "docx", "md"],
        accept_multiple_files=True,
        key="file_uploader",
        disabled=not kb_ready,
        help="Files will be parsed, chunked, and embedded into the active FAISS index.",
    )

    if st.button(
        "Index Uploaded Documents",
        key="btn_process_uploads",
        use_container_width=True,
        disabled=(not kb_ready or not uploaded_files),
    ):
        total_chunks = 0
        processed_names = []
        errors = []

        with st.status("Indexing uploaded files...", expanded=True) as upload_status:
            try:
                embeddings = rag.get_embeddings()
            except Exception as e:
                upload_status.update(label=f"Embedding error: {e}", state="error")
                st.stop()

            for uf in uploaded_files:
                st.write(f"Processing `{uf.name}`...")
                try:
                    docs = rag.load_uploaded_file(uf)
                    if not docs:
                        st.write(f"⚠️ No text found in `{uf.name}` — skipped.")
                        continue
                    vs, n_chunks = rag.add_documents_to_vectorstore(
                        docs,
                        st.session_state.vectorstore,
                        embeddings,
                    )
                    st.session_state.vectorstore = vs
                    total_chunks += n_chunks
                    processed_names.append(uf.name)
                    st.write(f"✅ Added {n_chunks} chunk(s) from `{uf.name}`")
                except Exception as e:
                    errors.append(f"{uf.name}: {e}")
                    st.write(f"❌ Failed `{uf.name}`: {e}")

        if processed_names:
            existing = st.session_state.doc_stats.get("files", [])
            new_files = [n for n in processed_names if n not in existing]
            st.session_state.doc_stats["files"] = existing + new_files
            st.session_state.doc_stats["file_count"] = len(st.session_state.doc_stats["files"])
            success_str = f"Successfully indexed {total_chunks} chunk(s) from {len(processed_names)} file(s)."
            upload_status.update(label=success_str, state="complete", expanded=False)
            st.session_state.upload_success_msg = success_str
            st.session_state.uploaded_file_names = processed_names
        else:
            upload_status.update(label="No files were successfully processed.", state="error", expanded=True)

        st.rerun()

    if st.session_state.upload_success_msg:
        st.success(st.session_state.upload_success_msg)
        st.session_state.upload_success_msg = ""

    st.markdown("---")

    # ── Index Management ──
    st.markdown("#### Management")
    c_btn1, c_btn2 = st.columns(2)
    with c_btn1:
        if st.button("🔄 Rebuild", key="btn_rebuild", use_container_width=True, help="Re-index all documents from scratch"):
            _init_pipeline.clear()
            import shutil
            if rag.FAISS_INDEX_PATH.exists():
                shutil.rmtree(str(rag.FAISS_INDEX_PATH))
            st.session_state.vectorstore = None
            st.session_state.llm = None
            st.session_state.kb_status = "not_loaded"
            st.session_state.doc_stats = {}
            st.session_state.chat_history = []
            st.rerun()
    with c_btn2:
        if st.button("🗑️ Clear", key="btn_clear", use_container_width=True, help="Clear conversation history"):
            st.session_state.chat_history = []
            st.rerun()

    st.markdown("---")

    # ── Example Queries ──
    st.markdown("#### Sample Questions")
    example_questions = [
        "What products does Nexora offer and what are their starting prices?",
        "How much does Nexora Flow Business plan cost and what does it include?",
        "What are the leave policies and annual leave entitlements at Nexora?",
        "What is the typical engagement duration for Cloud Modernization services?",
        "Does Nexora offer a free trial?",
    ]
    for eq in example_questions:
        if st.button(f"↗ {eq[:42]}...", key=f"ex_{hash(eq)}", use_container_width=True):
            st.session_state["prefill_question"] = eq
            st.rerun()

    st.markdown("---")

    # ── System Specifications ──
    st.markdown("#### Architecture Specs")
    st.markdown(
        f"""
        <div style="font-size:0.75rem; color:#64748B; line-height:1.6;">
        • <strong>Embeddings:</strong> <code>all-MiniLM-L6-v2</code><br/>
        • <strong>Vector DB:</strong> FAISS (L2 / Cosine)<br/>
        • <strong>Model:</strong> <code>{rag.LLM_MODEL}</code><br/>
        • <strong>Chunking:</strong> {rag.CHUNK_SIZE}c / {rag.CHUNK_OVERLAP}c overlap
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main Content Area: Corporate Layout
# ---------------------------------------------------------------------------

# ── Corporate Brand Header ──
st.markdown(
    """
    <div class="enterprise-header">
        <span class="badge-corporate">Verified Corporate Knowledge Base</span>
        <h1 class="main-title">Nexora Technologies Knowledge Assistant</h1>
        <p class="main-subtitle">
            Securely query organizational policies, product portfolios, services, pricing matrices, and technical SLAs.
            Answers are strictly grounded in verified internal documents with zero external hallucinations.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Auto-initialize Pipeline from Cache ──
if st.session_state.kb_status != "ready":
    try:
        _embeddings, _vs, _llm = _init_pipeline()
        st.session_state.vectorstore = _vs
        st.session_state.llm = _llm
        st.session_state.kb_status = "ready"
        st.session_state.kb_error = ""
        md_files = list(rag.DATA_DIR.glob("*.md"))
        st.session_state.doc_stats = {
            "file_count": len(md_files),
            "files": [f.name for f in md_files],
        }
        st.rerun()
    except Exception as e:
        st.session_state.kb_status = "error"
        st.session_state.kb_error = str(e)
        st.error(f"Failed to initialize knowledge base: {e}")
        st.stop()


# ---------------------------------------------------------------------------
# Helper: Render Conversation Turn with 2-Column Citations Grid
# ---------------------------------------------------------------------------
def render_turn(turn: dict):
    """
    Renders a single user-bot conversational exchange using structured layout:
    - Clean user question card
    - Grounded assistant response
    - 2-Column responsive citation card grid
    - Collapsible raw retrieved context (debug)
    """
    # 1. User Message
    st.markdown(
        f'<div class="chat-card-user"><strong>User:</strong> {turn["question"]}</div>',
        unsafe_allow_html=True,
    )

    # 2. Assistant Response
    answer = turn["answer"]
    if is_no_info_response(answer):
        st.markdown(
            f'<div class="alert-no-info">⚠️ <span>{rag.NO_INFO_RESPONSE}</span></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="chat-card-bot">{answer}</div>',
            unsafe_allow_html=True,
        )

    # 3. Citation Cards in 2-Column Layout
    sources = turn.get("sources", [])
    if sources and not is_no_info_response(answer):
        with st.expander(f"📎 Verified Sources ({len(sources)} references)", expanded=False):
            cols = st.columns(min(len(sources), 2))
            for idx, src in enumerate(sources):
                with cols[idx % 2]:
                    st.markdown(
                        f"""
                        <div class="source-card">
                            <div class="source-header">
                                <span class="source-tag">📄 {src["file"]}</span>
                                <span class="chunk-tag">Chunk #{src["chunk"]}</span>
                            </div>
                            <div class="source-title">{src["title"]}</div>
                            <div class="source-preview">"{src["preview"]}"</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    # 4. Raw Retrieved Context in Collapsible Expander
    if turn.get("context_text"):
        with st.expander("🔍 Raw Retrieved Context (Debug)", expanded=False):
            st.code(turn["context_text"], language="markdown")


# ---------------------------------------------------------------------------
# Chat Query Handling & Execution
# ---------------------------------------------------------------------------
query_to_run = None

# Check if an example query was triggered from the sidebar
if "prefill_question" in st.session_state and st.session_state["prefill_question"]:
    query_to_run = st.session_state.pop("prefill_question")

# Chat input container (pinned at bottom)
chat_input_val = st.chat_input("Ask a question about Nexora Technologies...")
if chat_input_val and chat_input_val.strip():
    query_to_run = chat_input_val.strip()

# ── Main Conversation Container ──
chat_container = st.container()

with chat_container:
    # Render all past turns
    if st.session_state.chat_history:
        for turn in st.session_state.chat_history:
            render_turn(turn)

    # If a new query is submitted: display user question immediately on screen
    if query_to_run:
        # 1. Immediately render user question card
        st.markdown(
            f'<div class="chat-card-user"><strong>User:</strong> {query_to_run}</div>',
            unsafe_allow_html=True,
        )

        # 2. Progress spinner directly under question
        if st.session_state.vectorstore is None or st.session_state.llm is None:
            st.error("Knowledge base is offline. Please click Rebuild Index in the sidebar.")
        else:
            with st.spinner("Searching internal documents and formulating grounded answer..."):
                try:
                    result = rag.answer_question(
                        query=query_to_run,
                        vectorstore=st.session_state.vectorstore,
                        llm=st.session_state.llm,
                    )
                    st.session_state.chat_history.append(
                        {
                            "question": query_to_run,
                            "answer": result["answer"],
                            "sources": result["sources"],
                            "context_text": result["context_text"],
                        }
                    )
                except Exception as e:
                    st.error(f"Error processing query: {e}")
                    st.code(traceback.format_exc(), language="python")
            st.rerun()

    # Empty State Display
    if not st.session_state.chat_history and not query_to_run and st.session_state.kb_status == "ready":
        st.markdown(
            """
            <div style="background-color: #111827; border: 1px dashed #334155; border-radius: 8px; padding: 48px 24px; text-align: center; margin-top: 16px;">
                <div style="font-size: 2.5rem; margin-bottom: 12px;">💬</div>
                <div style="font-size: 1.1rem; font-weight: 600; color: #F8FAFC; margin-bottom: 6px;">Knowledge Base Assistant Ready</div>
                <div style="font-size: 0.88rem; color: #94A3B8; max-width: 480px; margin: 0 auto 16px auto;">
                    Submit a question via the input bar below or select one of the sample inquiries from the sidebar to inspect verified company documentation.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
