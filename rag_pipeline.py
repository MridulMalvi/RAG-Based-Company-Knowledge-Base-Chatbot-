"""
rag_pipeline.py — Core RAG Pipeline for NovaTech Solutions Knowledge Base Chatbot
====================================================================================
Architecture:
  1. Document Loading  : Read .md files from data/ directory
  2. Text Splitting    : Chunk documents with overlap for context continuity
  3. Embedding         : Generate embeddings via HuggingFace sentence-transformers
  4. FAISS Index       : Store and persist embeddings for fast similarity search
  5. Retrieval         : Convert query to embedding → similarity search → top-k chunks
  6. LLM Generation   : Send retrieved context + question to OpenRouter LLM
  7. Anti-hallucination: System prompt strictly limits answers to retrieved context
"""

import os
import re
import logging
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = Path("data")
FAISS_INDEX_PATH = Path("faiss_index")

# Chunking parameters
CHUNK_SIZE = 800          # characters per chunk
CHUNK_OVERLAP = 150       # overlap between consecutive chunks

# Retrieval parameters
TOP_K_RESULTS = 5         # number of chunks to retrieve per query

# Embedding model (free, runs locally via sentence-transformers)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# LLM configuration (OpenRouter)
def get_openrouter_api_key() -> str:
    """Retrieve OpenRouter API key from Streamlit secrets or environment variables."""
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "OPENROUTER_API_KEY" in st.secrets:
            return str(st.secrets["OPENROUTER_API_KEY"]).strip()
    except Exception:
        pass
    return os.getenv("OPENROUTER_API_KEY", "").strip()


def get_openrouter_model() -> str:
    """Retrieve OpenRouter Model from Streamlit secrets or environment variables."""
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "OPENROUTER_MODEL" in st.secrets:
            return str(st.secrets["OPENROUTER_MODEL"]).strip()
    except Exception:
        pass
    return os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3.5-lightning:free").strip()


OPENROUTER_API_KEY = get_openrouter_api_key()
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = get_openrouter_model()

# Fallback sentinel response
NO_INFO_RESPONSE = "Information not available in knowledge base."

# System prompt enforcing strict grounding
SYSTEM_PROMPT = """You are the official AI knowledge assistant for Nexora Technologies Pvt. Ltd.
Your answers must be accurate and grounded in the context provided below (which includes Nexora company knowledge base documents and any user-uploaded files).

Guidelines:
1. Always interpret queries about "the company", "this company", "we", "our", "organization", "policies", or "products" as referring to Nexora Technologies (or any uploaded documents in context).
2. Answer clearly, accurately, and professionally based on the provided context.
3. If the context really does not contain the information needed to answer, reply with:
   "Information not available in knowledge base."
4. Deliver your direct, well-formatted answer immediately. Do not include any internal thought processes, planning commentary, or meta-notes.

Context:
{context}"""


def clean_llm_response(text: str) -> str:
    """
    Strips internal thinking/reasoning tags, echoed context sections,
    and scratchpad/verification notes so only the clean final answer is displayed.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    cleaned = text

    # 1. Strip XML-like thinking/thought/reasoning tags
    cleaned = re.sub(r"<(think|thought|reasoning)>.*?</\1>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<(think|thought|reasoning)>.*", "", cleaned, flags=re.DOTALL | re.IGNORECASE)

    # 2. Strip BBCode-style tags: [THINK]...[/THINK]
    cleaned = re.sub(r"\[(THINK|THOUGHT|REASONING)\].*?\[/\1\]", "", cleaned, flags=re.DOTALL | re.IGNORECASE)

    # 3. Detect known Nemotron / reasoning scratchpad markers
    scratchpad_terms = [
        "check constraints",
        "i'll draft",
        "i will draft",
        "verify against context",
        "verification against context",
        "chunking",
        "thinking process",
        "thought process",
        "reasoning steps",
        "scratchpad",
    ]
    has_scratchpad = any(marker in cleaned.lower() for marker in scratchpad_terms)

    if has_scratchpad:
        # Check if there is an explicit answer header
        splits = re.split(
            r"(?i)\n+(?:#{1,4}\s*|\*{1,2})?(?:Final Answer|Final Response|Answer|Response|Direct Answer)(?:\*{1,2})?:?\s*\n*",
            cleaned,
        )
        if len(splits) > 1 and splits[-1].strip():
            cleaned = splits[-1]
        else:
            # Strip scratchpad sections sequentially
            # 1. Remove "Chunking..." block
            cleaned = re.sub(r"(?is)(?:#{1,4}\s*|\*{1,2})?chunking.*?(?=\n\n|\Z)", "", cleaned)
            # 2. Remove "Check constraints..." block
            cleaned = re.sub(r"(?is)(?:#{1,4}\s*|\*{1,2})?check constraints.*?(?=\n\n(?:#{1,4}\s*|\*{1,2})?(?:i['’]ll draft|draft|verify|answer)|\Z)", "", cleaned)
            # 3. Remove "Verify against context..." block
            cleaned = re.sub(r"(?is)(?:#{1,4}\s*|\*{1,2})?verify against context.*?(?=\n\n|\Z)", "", cleaned)
            # 4. Remove leading "I'll draft..." header
            cleaned = re.sub(r"(?is)^.*?(?:i['’]ll draft|i will draft|drafting response):?\s*", "", cleaned)

    # 4. Remove leading lines echoing "Context:" or "Retrieved Context:"
    cleaned = re.sub(r"(?i)^(?:retrieved\s+)?context(?:\s+received)?:\s*.*?(?=\n\n|\r\n\r\n|$)", "", cleaned, flags=re.DOTALL)

    # 5. Clean up any leftover leading headers
    cleaned = cleaned.strip()
    cleaned = re.sub(
        r"^(?:#{1,4}\s*|\*{1,2})?(?:Final Answer|Final Response|Answer|Draft|Response)(?:\*{1,2})?:?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    res = cleaned.strip()

    # 6. Deduplicate repetitive degenerate loops (where the model repeats identical paragraphs or lists)
    blocks = re.split(r"\n{2,}", res)
    if len(blocks) > 1:
        unique_blocks = []
        seen_signatures = []
        for b in blocks:
            words = set(re.findall(r"[a-zA-Z0-9]{3,}", b.lower()))
            if len(words) >= 4:
                is_dup = False
                for prev_words in seen_signatures:
                    intersection = words & prev_words
                    union = words | prev_words
                    similarity = len(intersection) / len(union) if union else 0
                    if similarity > 0.60:
                        is_dup = True
                        break
                if is_dup:
                    break
                seen_signatures.append(words)
            unique_blocks.append(b)

        res = "\n\n".join(unique_blocks).strip()
        res = re.sub(r"(?i)\n+\s*(?:#{1,4}\s*|\*{1,2})?[a-zA-Z\s]{2,35}:(?:\*{1,2})?\s*$", "", res).strip()

    return res if res else text.strip()


# ---------------------------------------------------------------------------
# 1. Document Loading
# ---------------------------------------------------------------------------

def load_documents(data_dir: Path = DATA_DIR) -> list[Document]:
    """
    Load all supported files (.md, .pdf, .docx, .txt) from the data directory.
    Returns a list of LangChain Document objects with source metadata.
    """
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory '{data_dir}' not found.")

    supported_extensions = {".md", ".pdf", ".docx", ".txt"}
    all_files = [
        f for f in data_dir.iterdir()
        if f.is_file() and f.suffix.lower() in supported_extensions
    ]
    if not all_files:
        raise ValueError(f"No supported documents found in '{data_dir}'.")

    documents = []
    for file_path in all_files:
        try:
            ext = file_path.suffix.lower()
            if ext == ".pdf":
                loader = PyPDFLoader(str(file_path))
                docs = loader.load()
            elif ext == ".docx":
                loader = Docx2txtLoader(str(file_path))
                docs = loader.load()
            elif ext in (".txt", ".md"):
                try:
                    loader = TextLoader(str(file_path), encoding="utf-8")
                    docs = loader.load()
                except Exception:
                    loader = TextLoader(str(file_path), encoding="latin-1")
                    docs = loader.load()
            else:
                continue

            for doc in docs:
                doc.metadata["source"] = file_path.name
                doc.metadata["source_path"] = str(file_path)
                doc.metadata["doc_title"] = (
                    _extract_title(doc.page_content, file_path.stem)
                    if ext == ".md"
                    else file_path.stem.replace("_", " ").title()
                )
            documents.extend(docs)
            logger.info(f"Loaded: {file_path.name} ({len(docs)} document(s))")
        except Exception as e:
            logger.error(f"Failed to load {file_path.name}: {e}")

    logger.info(f"Total documents loaded: {len(documents)}")
    return documents


def _extract_title(content: str, fallback: str) -> str:
    """Extract the first H1 title from markdown content, or use filename as fallback."""
    match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return fallback.replace("_", " ").title()


# ---------------------------------------------------------------------------
# 2. Text Splitting
# ---------------------------------------------------------------------------

def split_documents(documents: list[Document]) -> list[Document]:
    """
    Split documents into chunks using RecursiveCharacterTextSplitter.
    Preserves source metadata in each chunk for citation purposes.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""],  # respect markdown structure
    )

    chunks = splitter.split_documents(documents)

    # Add chunk index to metadata for citation granularity
    source_counters: dict[str, int] = {}
    for chunk in chunks:
        src = chunk.metadata.get("source", "unknown")
        source_counters[src] = source_counters.get(src, 0) + 1
        chunk.metadata["chunk_index"] = source_counters[src]

    logger.info(f"Total chunks created: {len(chunks)}")
    return chunks


# ---------------------------------------------------------------------------
# 3. Embeddings
# ---------------------------------------------------------------------------

def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Return a HuggingFace embedding model.
    Downloads model on first run; cached locally thereafter.
    """
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return embeddings


# ---------------------------------------------------------------------------
# 4. FAISS Vector Store
# ---------------------------------------------------------------------------

def build_vectorstore(chunks: list[Document], embeddings: HuggingFaceEmbeddings) -> FAISS:
    """
    Build a FAISS vector store from document chunks and persist it to disk.
    """
    logger.info("Building FAISS vector store...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(str(FAISS_INDEX_PATH))
    logger.info(f"FAISS index saved to: {FAISS_INDEX_PATH}")
    return vectorstore


def load_vectorstore(embeddings: HuggingFaceEmbeddings) -> Optional[FAISS]:
    """
    Load a previously persisted FAISS vector store from disk.
    Returns None if the index does not exist.
    """
    if FAISS_INDEX_PATH.exists():
        logger.info(f"Loading FAISS index from: {FAISS_INDEX_PATH}")
        vectorstore = FAISS.load_local(
            str(FAISS_INDEX_PATH),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        return vectorstore
    return None


# ---------------------------------------------------------------------------
# 4b. Upload-based Document Loading
# ---------------------------------------------------------------------------

def load_uploaded_file(uploaded_file) -> list[Document]:
    """
    Parse an in-memory Streamlit UploadedFile object into LangChain Documents.
    Supported extensions: .pdf, .txt, .md, .docx
    """
    import tempfile

    file_name = uploaded_file.name
    ext = Path(file_name).suffix.lower()

    if hasattr(uploaded_file, "seek"):
        uploaded_file.seek(0)
    file_bytes = uploaded_file.read()
    if hasattr(uploaded_file, "seek"):
        uploaded_file.seek(0)

    if not file_bytes:
        logger.warning(f"Uploaded file '{file_name}' is empty.")
        return []

    # Write to a named temp file so path-based loaders can open it
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    docs = []
    try:
        if ext == ".pdf":
            try:
                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
            except Exception as pdf_err:
                logger.warning(f"PyPDFLoader failed ({pdf_err}), falling back to pypdf.PdfReader...")
                import pypdf
                reader = pypdf.PdfReader(tmp_path)
                for i, page in enumerate(reader.pages):
                    text = page.extract_text() or ""
                    if text.strip():
                        docs.append(Document(page_content=text, metadata={"page": i}))
        elif ext == ".docx":
            try:
                loader = Docx2txtLoader(tmp_path)
                docs = loader.load()
            except Exception as docx_err:
                logger.warning(f"Docx2txtLoader failed ({docx_err}), falling back to docx2txt...")
                import docx2txt
                text = docx2txt.process(tmp_path) or ""
                if text.strip():
                    docs.append(Document(page_content=text))
        elif ext in (".txt", ".md"):
            try:
                loader = TextLoader(tmp_path, encoding="utf-8")
                docs = loader.load()
            except Exception:
                loader = TextLoader(tmp_path, encoding="latin-1")
                docs = loader.load()
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)   # always clean up temp file

    # Filter out empty pages
    docs = [d for d in docs if d.page_content and d.page_content.strip()]

    # Normalise metadata: attach the original filename and a human title
    for doc in docs:
        doc.metadata["source"] = file_name
        doc.metadata["source_path"] = file_name
        doc.metadata["doc_title"] = (
            _extract_title(doc.page_content, Path(file_name).stem)
            if ext == ".md"
            else Path(file_name).stem.replace("_", " ").title()
        )

    logger.info(f"Loaded uploaded file '{file_name}': {len(docs)} page(s)/doc(s)")
    return docs


def add_documents_to_vectorstore(
    new_docs: list[Document],
    vectorstore: FAISS,
    embeddings: HuggingFaceEmbeddings,
) -> tuple[FAISS, int]:
    """
    Chunk new documents and merge them into the existing in-memory FAISS index.
    Persists the updated index back to disk so it survives page refreshes.

    Returns:
        (updated_vectorstore, num_new_chunks)
    """
    if not new_docs:
        return vectorstore, 0

    chunks = split_documents(new_docs)
    if not chunks:
        return vectorstore, 0

    # Merge into the live FAISS index (in-place, no rebuild needed)
    vectorstore.add_documents(chunks)

    # Persist so the enriched index is reused on next Load
    vectorstore.save_local(str(FAISS_INDEX_PATH))
    logger.info(
        f"Added {len(chunks)} chunk(s) from {len(new_docs)} page(s) to FAISS index."
    )
    return vectorstore, len(chunks)


# ---------------------------------------------------------------------------
# 5. Retrieval
# ---------------------------------------------------------------------------

def retrieve_context(query: str, vectorstore: FAISS, k: int = TOP_K_RESULTS) -> list[Document]:
    """
    Convert a user query to an embedding and retrieve the top-k most similar
    document chunks from the FAISS vector store.
    """
    results = vectorstore.similarity_search(query, k=k)
    logger.info(f"Retrieved {len(results)} chunk(s) for query: '{query[:60]}...'")
    return results


def format_context_for_prompt(retrieved_docs: list[Document]) -> str:
    """
    Format retrieved document chunks into a single context string for the LLM prompt.
    Each chunk is clearly labelled with its source file and chunk index.
    """
    context_parts = []
    for i, doc in enumerate(retrieved_docs, start=1):
        source = doc.metadata.get("source", "Unknown")
        chunk_idx = doc.metadata.get("chunk_index", "?")
        title = doc.metadata.get("doc_title", source)
        header = f"[Source {i}: {title} | File: {source} | Chunk #{chunk_idx}]"
        context_parts.append(f"{header}\n{doc.page_content.strip()}")

    return "\n\n---\n\n".join(context_parts)


# ---------------------------------------------------------------------------
# 6. LLM (OpenRouter)
# ---------------------------------------------------------------------------

def get_llm() -> ChatOpenAI:
    """
    Instantiate the ChatOpenAI client pointing to the OpenRouter API endpoint.
    """
    api_key = get_openrouter_api_key()
    if not api_key or api_key == "your_openrouter_api_key_here":
        raise ValueError(
            "OPENROUTER_API_KEY is not set. Please add it to your Streamlit Secrets or .env file."
        )

    model_name = get_openrouter_model()
    llm = ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        temperature=0.2,          # 0.2 prevents deterministic greedy repetition loops
        frequency_penalty=0.5,    # prevents repeating same paragraphs or lists
        presence_penalty=0.3,
        max_tokens=1024,
        default_headers={
            "HTTP-Referer": "https://nexora.tech",
            "X-Title": "Nexora KB Chatbot",
        },
    )
    return llm


# ---------------------------------------------------------------------------
# 7. End-to-End Query Handler
# ---------------------------------------------------------------------------

def answer_question(
    query: str,
    vectorstore: FAISS,
    llm: ChatOpenAI,
) -> dict:
    """
    Full RAG pipeline for a single question:
      1. Retrieve relevant chunks
      2. Format context
      3. Call LLM with grounded system prompt
      4. Return structured result with answer + citations

    Returns:
        dict with keys:
          - "answer"       : str  — LLM-generated answer
          - "sources"      : list — List of citation dicts
          - "context_text" : str  — Raw retrieved context (for display)
          - "query"        : str  — Original user query
    """
    # Step 1: Retrieve context
    retrieved_docs = retrieve_context(query, vectorstore)

    # Step 2: Format context string
    context_text = format_context_for_prompt(retrieved_docs)

    # Step 3: Build messages for LLM
    system_content = SYSTEM_PROMPT.format(context=context_text)
    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=query),
    ]

    # Step 4: Call LLM
    try:
        response = llm.invoke(messages)
        raw_answer = response.content.strip() if hasattr(response, "content") else str(response).strip()
        answer = clean_llm_response(raw_answer)
        if not answer:
            answer = raw_answer
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        answer = f"Error communicating with the LLM: {e}"

    # Step 5: Build citations
    sources = []
    seen = set()
    for doc in retrieved_docs:
        src = doc.metadata.get("source", "Unknown")
        chunk_idx = doc.metadata.get("chunk_index", "?")
        title = doc.metadata.get("doc_title", src)
        citation_key = f"{src}::{chunk_idx}"
        if citation_key not in seen:
            seen.add(citation_key)
            sources.append({
                "file": src,
                "title": title,
                "chunk": chunk_idx,
                "preview": doc.page_content[:200].strip() + "...",
            })

    return {
        "query": query,
        "answer": answer,
        "sources": sources,
        "context_text": context_text,
    }


# ---------------------------------------------------------------------------
# 8. Pipeline Initializer (convenience wrapper)
# ---------------------------------------------------------------------------

def initialize_pipeline(force_rebuild: bool = False) -> tuple[FAISS, ChatOpenAI]:
    """
    Convenience function to initialize the full RAG pipeline:
      - Loads or builds the FAISS vector store
      - Returns (vectorstore, llm) ready for querying

    Args:
        force_rebuild: If True, always rebuild the FAISS index from source documents.
    """
    embeddings = get_embeddings()

    if not force_rebuild:
        vectorstore = load_vectorstore(embeddings)
        if vectorstore:
            logger.info("Using cached FAISS index.")
            llm = get_llm()
            return vectorstore, llm

    # Build from scratch
    logger.info("Building vector store from source documents...")
    documents = load_documents()
    chunks = split_documents(documents)
    vectorstore = build_vectorstore(chunks, embeddings)
    llm = get_llm()
    return vectorstore, llm


# ---------------------------------------------------------------------------
# CLI Quick Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("NovaTech KB Chatbot — RAG Pipeline Quick Test")
    print("=" * 60)

    vs, llm_client = initialize_pipeline()

    test_query = "What is NovaDrive and what are its key features?"
    print(f"\nQuery: {test_query}\n")

    result = answer_question(test_query, vs, llm_client)

    print("Answer:")
    print(result["answer"])
    print("\nSources:")
    for s in result["sources"]:
        print(f"  - {s['title']} | {s['file']} | Chunk #{s['chunk']}")
