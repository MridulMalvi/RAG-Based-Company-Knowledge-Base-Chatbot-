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
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# Read model from .env (OPENROUTER_MODEL), fall back to an active free model
LLM_MODEL = os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3.5-lightning:free")

# Fallback sentinel response
NO_INFO_RESPONSE = "Information not available in knowledge base."

# System prompt enforcing strict grounding
SYSTEM_PROMPT = """You are a helpful assistant for Nexora Technologies, an IT company.
Your ONLY knowledge source is the context retrieved from the company knowledge base provided below.

STRICT RULES — YOU MUST FOLLOW THESE WITHOUT EXCEPTION:
1. Answer ONLY using the information present in the provided context.
2. Do NOT use any external knowledge, training data, or assumptions.
3. If the context does not contain enough information to answer the question, respond with EXACTLY this phrase and nothing else:
   "Information not available in knowledge base."
4. Do NOT guess, speculate, or provide partial answers from external knowledge.
5. Every factual statement must be directly traceable to the provided context.
6. Keep your answer clear, concise, and professional.

Context from Knowledge Base:
{context}

Remember: If the answer is not in the context above, say exactly "Information not available in knowledge base." """


# ---------------------------------------------------------------------------
# 1. Document Loading
# ---------------------------------------------------------------------------

def load_documents(data_dir: Path = DATA_DIR) -> list[Document]:
    """
    Load all Markdown (.md) files from the data directory.
    Returns a list of LangChain Document objects with source metadata.
    """
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory '{data_dir}' not found.")

    md_files = list(data_dir.glob("*.md"))
    if not md_files:
        raise ValueError(f"No .md files found in '{data_dir}'.")

    documents = []
    for file_path in md_files:
        try:
            loader = TextLoader(str(file_path), encoding="utf-8")
            docs = loader.load()
            # Enrich metadata with a clean, readable source name
            for doc in docs:
                doc.metadata["source"] = file_path.name
                doc.metadata["source_path"] = str(file_path)
                # Extract a friendly document title from the first H1 heading
                doc.metadata["doc_title"] = _extract_title(doc.page_content, file_path.stem)
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
    The file bytes are written to a temp file so the loaders (which expect a
    file path) can read them; the temp file is deleted immediately after.
    """
    import tempfile

    file_name = uploaded_file.name
    ext = Path(file_name).suffix.lower()
    file_bytes = uploaded_file.read()

    # Write to a named temp file so path-based loaders can open it
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        if ext == ".pdf":
            loader = PyPDFLoader(tmp_path)
        elif ext == ".docx":
            loader = Docx2txtLoader(tmp_path)
        elif ext in (".txt", ".md"):
            loader = TextLoader(tmp_path, encoding="utf-8")
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        docs = loader.load()
    finally:
        Path(tmp_path).unlink(missing_ok=True)   # always clean up temp file

    # Normalise metadata: attach the original filename and a human title
    for doc in docs:
        doc.metadata["source"] = file_name
        doc.metadata["source_path"] = file_name
        doc.metadata["doc_title"] = _extract_title(
            doc.page_content,
            Path(file_name).stem,
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
    if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "your_openrouter_api_key_here":
        raise ValueError(
            "OPENROUTER_API_KEY is not set. Please add it to your .env file."
        )

    llm = ChatOpenAI(
        model=LLM_MODEL,
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        temperature=0.1,          # low temperature for factual consistency
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
        answer = response.content.strip()
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
