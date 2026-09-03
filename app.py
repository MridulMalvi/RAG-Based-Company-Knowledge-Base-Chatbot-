"""
app.py — Streamlit UI for NovaTech Solutions Knowledge Base Chatbot
=====================================================================
Provides a polished, interactive chat interface powered by the RAG pipeline
defined in rag_pipeline.py.
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
    page_title="NovaTech KB Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — Modern Dark UI
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* ── Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Background ── */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        min-height: 100vh;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }

    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #a78bfa;
    }

    /* ── Main content text ── */
    .stMarkdown, .stText, p, li, label {
        color: #e2e8f0 !important;
    }

    /* ── Chat bubbles ── */
    .user-bubble {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 14px 18px;
        border-radius: 18px 18px 4px 18px;
        margin: 8px 0;
        max-width: 85%;
        margin-left: auto;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        font-size: 0.95rem;
        line-height: 1.6;
    }

    .bot-bubble {
        background: rgba(255, 255, 255, 0.07);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(167, 139, 250, 0.3);
        color: #e2e8f0;
        padding: 14px 18px;
        border-radius: 18px 18px 18px 4px;
        margin: 8px 0;
        max-width: 95%;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        font-size: 0.95rem;
        line-height: 1.6;
    }

    /* ── No-info warning ── */
    .no-info-box {
        background: rgba(251, 191, 36, 0.1);
        border: 1px solid rgba(251, 191, 36, 0.5);
        border-radius: 12px;
        padding: 14px 18px;
        color: #fbbf24;
        font-weight: 500;
    }

    /* ── Citation cards ── */
    .citation-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 10px;
        padding: 10px 14px;
        margin: 6px 0;
        font-size: 0.82rem;
        color: #a5b4fc;
    }
    .citation-card strong {
        color: #c4b5fd;
    }

    /* ── Context expander ── */
    .context-box {
        background: rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 12px 16px;
        font-size: 0.8rem;
        color: #94a3b8;
        font-family: 'Courier New', monospace;
        max-height: 300px;
        overflow-y: auto;
        white-space: pre-wrap;
    }

    /* ── Status badges ── */
    .badge-ready {
        background: rgba(34, 197, 94, 0.15);
        border: 1px solid rgba(34, 197, 94, 0.4);
        color: #4ade80;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .badge-not-ready {
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid rgba(239, 68, 68, 0.4);
        color: #f87171;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .badge-loading {
        background: rgba(234, 179, 8, 0.15);
        border: 1px solid rgba(234, 179, 8, 0.4);
        color: #facc15;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
    }

    /* ── Input box ── */
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.07) !important;
        border: 1px solid rgba(167, 139, 250, 0.4) !important;
        color: #e2e8f0 !important;
        border-radius: 12px !important;
        padding: 12px 16px !important;
        font-size: 0.95rem !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #a78bfa !important;
        box-shadow: 0 0 0 2px rgba(167, 139, 250, 0.2) !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.35) !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5) !important;
    }

    /* ── Divider ── */
    hr {
        border-color: rgba(255, 255, 255, 0.1) !important;
    }

    /* ── Title ── */
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #a78bfa, #60a5fa, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.2rem;
    }
    .main-subtitle {
        font-size: 0.95rem;
        color: #94a3b8;
        margin-bottom: 2rem;
    }

    /* ── Expander ── */
    details summary {
        color: #a78bfa !important;
        font-size: 0.85rem;
        cursor: pointer;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(167, 139, 250, 0.3); border-radius: 3px; }

    /* ── Hide Streamlit branding ── */
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
    st.session_state.kb_status = "not_loaded"   # not_loaded | loading | ready | error
if "kb_error" not in st.session_state:
    st.session_state.kb_error = ""
if "doc_stats" not in st.session_state:
    st.session_state.doc_stats = {}
if "upload_success_msg" not in st.session_state:
    st.session_state.upload_success_msg = ""
if "uploaded_file_names" not in st.session_state:
    st.session_state.uploaded_file_names = []


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def load_knowledge_base(force_rebuild: bool = False):
    """Initialize the RAG pipeline and store results in session state."""
    st.session_state.kb_status = "loading"
    st.session_state.kb_error = ""
    try:
        vectorstore, llm = rag.initialize_pipeline(force_rebuild=force_rebuild)
        st.session_state.vectorstore = vectorstore
        st.session_state.llm = llm
        st.session_state.kb_status = "ready"

        # Gather doc stats for display
        data_dir = rag.DATA_DIR
        md_files = list(data_dir.glob("*.md"))
        st.session_state.doc_stats = {
            "file_count": len(md_files),
            "files": [f.name for f in md_files],
        }
    except Exception as e:
        st.session_state.kb_status = "error"
        st.session_state.kb_error = str(e)
        st.session_state.vectorstore = None
        st.session_state.llm = None


def is_no_info_response(answer: str) -> bool:
    """Check if the LLM returned the 'not available' sentinel."""
    return rag.NO_INFO_RESPONSE.lower() in answer.lower()


def render_status_badge():
    """Render a colored status badge in the sidebar."""
    status = st.session_state.kb_status
    if status == "ready":
        st.markdown('<span class="badge-ready">● KB Ready</span>', unsafe_allow_html=True)
    elif status == "loading":
        st.markdown('<span class="badge-loading">⏳ Loading...</span>', unsafe_allow_html=True)
    elif status == "error":
        st.markdown('<span class="badge-not-ready">✗ Error</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-not-ready">○ Not Loaded</span>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🏢 NovaTech KB Chatbot")
    st.markdown("*Powered by RAG + OpenRouter*")
    st.markdown("---")

    # Status
    st.markdown("### 📊 Knowledge Base Status")
    render_status_badge()

    if st.session_state.kb_status == "ready":
        stats = st.session_state.doc_stats
        st.markdown(f"**Documents indexed:** {stats.get('file_count', 0)}")
        with st.expander("📁 Indexed files"):
            for fname in stats.get("files", []):
                nice = fname.replace(".md", "").replace("_", " ").title()
                st.markdown(f"• {nice}")

    elif st.session_state.kb_status == "error":
        st.error(f"Error: {st.session_state.kb_error}")

    st.markdown("---")

    # Actions
    st.markdown("### ⚙️ Actions")

    col1, col2 = st.columns(2)
    with col1:
        load_clicked = st.button("▶ Load KB", key="btn_load", use_container_width=True)
    with col2:
        rebuild_clicked = st.button("🔄 Rebuild", key="btn_rebuild", use_container_width=True)

    if load_clicked or rebuild_clicked:
        force = rebuild_clicked
        # Show live status right in the sidebar
        with st.status("⏳ Initializing...", expanded=True) as status_box:
            st.write("Loading embedding model (all-MiniLM-L6-v2)...")
            try:
                embeddings = rag.get_embeddings()
                st.write("✅ Embedding model ready")
                if force or not rag.FAISS_INDEX_PATH.exists():
                    st.write("📄 Loading & chunking documents...")
                    docs = rag.load_documents()
                    chunks = rag.split_documents(docs)
                    st.write(f"✅ {len(chunks)} chunks created from {len(docs)} documents")
                    st.write("🔢 Building FAISS index...")
                    vectorstore = rag.build_vectorstore(chunks, embeddings)
                else:
                    st.write("📦 Loading cached FAISS index...")
                    vectorstore = rag.load_vectorstore(embeddings)
                st.write("✅ FAISS index ready")
                st.write("🤖 Connecting to OpenRouter LLM...")
                llm = rag.get_llm()
                st.write("✅ LLM connected")
                st.session_state.vectorstore = vectorstore
                st.session_state.llm = llm
                st.session_state.kb_status = "ready"
                st.session_state.kb_error = ""
                data_dir = rag.DATA_DIR
                md_files = list(data_dir.glob("*.md"))
                st.session_state.doc_stats = {
                    "file_count": len(md_files),
                    "files": [f.name for f in md_files],
                }
                status_box.update(label="✅ Knowledge Base Ready!", state="complete", expanded=False)
            except Exception as e:
                st.session_state.kb_status = "error"
                st.session_state.kb_error = str(e)
                st.session_state.vectorstore = None
                st.session_state.llm = None
                status_box.update(label=f"❌ Error: {e}", state="error", expanded=True)
        st.rerun()

    if st.button("🗑️ Clear Chat", key="btn_clear", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("---")

    # ── Upload Documents ──────────────────────────────────────────────────────
    st.markdown("### 📤 Upload Documents")

    kb_ready = st.session_state.kb_status == "ready"
    if not kb_ready:
        st.caption("⚠️ Load the Knowledge Base first before uploading.")

    uploaded_files = st.file_uploader(
        "Add files to the Knowledge Base",
        type=["pdf", "txt", "docx", "md"],
        accept_multiple_files=True,
        key="file_uploader",
        disabled=not kb_ready,
        help="Supported formats: PDF, TXT, DOCX, MD. Files will be chunked and added to the current FAISS index.",
    )

    if st.button(
        "⚡ Process Uploaded Docs",
        key="btn_process_uploads",
        use_container_width=True,
        disabled=(not kb_ready or not uploaded_files),
    ):
        total_chunks = 0
        processed_names = []
        errors = []

        with st.status("📥 Processing uploads...", expanded=True) as upload_status:
            # Re-acquire embeddings (already cached in memory by sentence-transformers)
            try:
                embeddings = rag.get_embeddings()
            except Exception as e:
                upload_status.update(label=f"❌ Embedding model error: {e}", state="error")
                st.stop()

            for uf in uploaded_files:
                st.write(f"📄 Processing `{uf.name}`…")
                try:
                    docs = rag.load_uploaded_file(uf)
                    if not docs:
                        st.write(f"  ⚠️ No text extracted from `{uf.name}` — skipped.")
                        continue
                    vs, n_chunks = rag.add_documents_to_vectorstore(
                        docs,
                        st.session_state.vectorstore,
                        embeddings,
                    )
                    st.session_state.vectorstore = vs
                    total_chunks += n_chunks
                    processed_names.append(uf.name)
                    st.write(f"  ✅ Added {n_chunks} chunk(s) from `{uf.name}`")
                except Exception as e:
                    errors.append(f"{uf.name}: {e}")
                    st.write(f"  ❌ Failed `{uf.name}`: {e}")

        if processed_names:
            # Update the doc stats so the sidebar counter refreshes
            existing = st.session_state.doc_stats.get("files", [])
            new_files = [n for n in processed_names if n not in existing]
            st.session_state.doc_stats["files"] = existing + new_files
            st.session_state.doc_stats["file_count"] = len(
                st.session_state.doc_stats["files"]
            )
            upload_label = (
                f"✅ Indexed {total_chunks} chunk(s) from "
                f"{len(processed_names)} file(s)!"
            )
            upload_status.update(label=upload_label, state="complete", expanded=False)
            st.session_state.upload_success_msg = upload_label
            st.session_state.uploaded_file_names = processed_names
        else:
            fail_label = "❌ No files were processed successfully."
            upload_status.update(label=fail_label, state="error", expanded=True)

        st.rerun()

    # Persist success message across reruns and clear it after display
    if st.session_state.upload_success_msg:
        st.success(st.session_state.upload_success_msg)
        st.session_state.upload_success_msg = ""  # show once then clear

    st.markdown("---")

    # Configuration info
    st.markdown("### 🔧 Configuration")
    st.markdown(f"**Embedding:** `all-MiniLM-L6-v2`")
    st.markdown(f"**LLM:** `{rag.LLM_MODEL}`")
    st.markdown(f"**Chunk size:** {rag.CHUNK_SIZE} chars")
    st.markdown(f"**Chunk overlap:** {rag.CHUNK_OVERLAP} chars")
    st.markdown(f"**Top-K retrieval:** {rag.TOP_K_RESULTS} chunks")

    st.markdown("---")

    # Example questions
    st.markdown("### 💡 Example Questions")
    example_questions = [
        "What products does Nexora offer and what are their starting prices?",
        "How much does Nexora Flow Business plan cost and what does it include?",
        "What are the leave policies and annual leave entitlements at Nexora?",
        "What is the typical engagement duration for Cloud Modernization services?",
        "Does Nexora offer a free trial?",
    ]
    for eq in example_questions:
        if st.button(f"↗ {eq[:40]}...", key=f"ex_{hash(eq)}", use_container_width=True):
            st.session_state["prefill_question"] = eq
            st.rerun()

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.75rem; color:#64748b; text-align:center;'>"
        "Nexora Technologies KB Chatbot v1.0<br/>Built with LangChain + FAISS + Streamlit"
        "</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main Content Area
# ---------------------------------------------------------------------------
st.markdown('<h1 class="main-title">🤖 Nexora Knowledge Base Chatbot</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="main-subtitle">Ask anything about Nexora Technologies — products, services, pricing, HR policies, and more.</p>',
    unsafe_allow_html=True,
)

# ── Welcome / Not-loaded state ──
if st.session_state.kb_status == "not_loaded":
    st.markdown(
        """
        <div style="text-align:center; padding: 60px 20px;">
            <div style="font-size: 5rem; margin-bottom: 16px;">🗄️</div>
            <div style="font-size: 1.4rem; font-weight: 600; color: #a78bfa; margin-bottom: 8px;">
                Knowledge Base Not Loaded
            </div>
            <div style="font-size: 0.95rem; color: #94a3b8; max-width: 500px; margin: 0 auto 28px auto;">
                Click <strong style="color:#c4b5fd;">▶ Load KB</strong> in the sidebar to initialize the
                embedding model and build the FAISS index from the 6 knowledge-base documents.
                This takes ~30–60 seconds on first run, then loads instantly from cache.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# ── Error state ──
if st.session_state.kb_status == "error":
    st.error(
        f"❌ **Failed to initialize the Knowledge Base**\n\n"
        f"**Error:** `{st.session_state.kb_error}`\n\n"
        "**Common fixes:**\n"
        "- Add your `OPENROUTER_API_KEY` to the `.env` file and click **▶ Load KB** again\n"
        "- Ensure the `data/` directory contains `.md` files\n"
        "- Run `python -m pip install -r requirements.txt` in your virtualenv"
    )
    st.stop()

# ── Chat History Display ──
if st.session_state.chat_history:
    st.markdown("### 💬 Conversation")
    for turn in st.session_state.chat_history:
        # User bubble
        st.markdown(
            f'<div class="user-bubble">👤 {turn["question"]}</div>',
            unsafe_allow_html=True,
        )

        # Bot bubble or no-info box
        answer = turn["answer"]
        if is_no_info_response(answer):
            st.markdown(
                f'<div class="no-info-box">⚠️ {rag.NO_INFO_RESPONSE}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="bot-bubble">🤖 {answer}</div>',
                unsafe_allow_html=True,
            )

        # Sources / Citations
        if turn.get("sources") and not is_no_info_response(answer):
            with st.expander(f"📎 Sources ({len(turn['sources'])} referenced)", expanded=False):
                for src in turn["sources"]:
                    st.markdown(
                        f'<div class="citation-card">'
                        f'<strong>📄 {src["title"]}</strong><br/>'
                        f'File: <code>{src["file"]}</code> | Chunk #<code>{src["chunk"]}</code><br/>'
                        f'<em>Preview:</em> {src["preview"]}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

        # Retrieved context (collapsed by default)
        if turn.get("context_text"):
            with st.expander("🔍 Retrieved Context (raw)", expanded=False):
                st.markdown(
                    f'<div class="context-box">{turn["context_text"]}</div>',
                    unsafe_allow_html=True,
                )

        st.markdown("<hr/>", unsafe_allow_html=True)

# ── Input Area ──
st.markdown("### ✍️ Ask a Question")

# Handle pre-filled question from sidebar example buttons
prefill = st.session_state.pop("prefill_question", "")

with st.form(key="question_form", clear_on_submit=True):
    user_question = st.text_input(
        label="Your question",
        value=prefill,
        placeholder="e.g. What are the pricing tiers for NovaDrive?",
        label_visibility="collapsed",
    )
    col_btn1, col_btn2 = st.columns([1, 5])
    with col_btn1:
        submitted = st.form_submit_button("Ask →", use_container_width=True)

if submitted and user_question.strip():
    if st.session_state.vectorstore is None or st.session_state.llm is None:
        st.error("Knowledge base is not loaded. Click 'Load KB' in the sidebar.")
    else:
        with st.spinner("🔎 Retrieving context and generating answer..."):
            try:
                result = rag.answer_question(
                    query=user_question.strip(),
                    vectorstore=st.session_state.vectorstore,
                    llm=st.session_state.llm,
                )
                # Append to chat history
                st.session_state.chat_history.append(
                    {
                        "question": user_question.strip(),
                        "answer": result["answer"],
                        "sources": result["sources"],
                        "context_text": result["context_text"],
                    }
                )
            except Exception as e:
                st.error(f"An error occurred while processing your query:\n\n`{e}`")
                st.markdown(f"```\n{traceback.format_exc()}\n```")
        st.rerun()

elif submitted and not user_question.strip():
    st.warning("Please enter a question before submitting.")

# ── Empty state placeholder ──
if not st.session_state.chat_history and st.session_state.kb_status == "ready":
    st.markdown(
        """
        <div style="text-align:center; padding: 60px 20px; color: #64748b;">
            <div style="font-size: 4rem; margin-bottom: 16px;">💬</div>
            <div style="font-size: 1.1rem; font-weight: 500; color: #94a3b8;">Ready to answer your questions!</div>
            <div style="font-size: 0.9rem; margin-top: 8px;">
                Type a question above or pick one from the sidebar examples.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
