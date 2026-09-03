# 🤖 Nexora Technologies — RAG-Based Company Knowledge Base Chatbot

A production-quality, **Retrieval-Augmented Generation (RAG)** chatbot that lets anyone query the internal knowledge base of *Nexora Technologies* (a fictional IT company) using natural language — and receive grounded, cited answers.

> **Zero hallucinations.** If the answer isn't in the knowledge base, the chatbot says so explicitly.

---

## 📸 Features

| Feature | Details |
|---|---|
| **RAG Architecture** | FAISS vector search + LangChain orchestration |
| **Anti-Hallucination** | Strict system prompt limits answers to retrieved context only |
| **Citations** | Every answer links back to the source file and chunk number |
| **Free Embeddings** | `all-MiniLM-L6-v2` via `sentence-transformers` (runs locally, no API cost) |
| **Free LLM** | OpenRouter API with active free models (e.g., `nvidia/nemotron-3.5-lightning:free`) |
| **Modern UI** | Streamlit with dark glassmorphism theme, chat history, and context viewer |
| **Persistent Index** | FAISS index is saved to disk and reused across sessions |
| **Evaluation Suite** | 22-question test script with JSON + text report output |

---

## 🏗️ Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                       User Interface (Streamlit)              │
│   Sidebar: KB Load / Rebuild / Status                         │
│   Main: Chat Input → Answer + Citations + Context Viewer      │
└───────────────────────┬───────────────────────────────────────┘
                        │ question
                        ▼
┌───────────────────────────────────────────────────────────────┐
│                    RAG Pipeline (rag_pipeline.py)             │
│                                                               │
│  1. QUERY EMBEDDING                                           │
│     sentence-transformers/all-MiniLM-L6-v2                   │
│     question → 384-dim dense vector                           │
│                                                               │
│  2. SIMILARITY SEARCH                                         │
│     FAISS IndexFlatIP (cosine similarity via L2 norm)         │
│     → Top-5 most relevant document chunks                     │
│                                                               │
│  3. CONTEXT FORMATTING                                        │
│     Chunks labelled with source file + chunk index            │
│                                                               │
│  4. LLM GENERATION (OpenRouter)                               │
│     SystemPrompt [strict grounding rules + context]           │
│     + HumanMessage [user question]                            │
│     → meta-llama/llama-3.3-70b-instruct:free                 │
│     → Grounded answer (or "Information not available...")     │
│                                                               │
│  5. CITATION EXTRACTION                                       │
│     Source file, title, chunk index, preview for each chunk   │
└───────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────────────────────┐
│                    Knowledge Base (data/)                     │
│   company_profile.md  │  products.md  │  services.md         │
│   pricing.md          │  hr_policies.md  │  faqs.md           │
│                                                               │
│   Indexed as FAISS vector store (faiss_index/)               │
│   Chunk size: 800 chars | Overlap: 150 chars                  │
└───────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

**Why FAISS over a hosted vector DB?**  
FAISS runs completely locally, requires no API key, is fast for datasets of this size, and the index persists across app restarts — ideal for a self-contained demo.

**Why HuggingFace embeddings?**  
`all-MiniLM-L6-v2` produces high-quality 384-dimensional embeddings, is freely available, runs on CPU, and has excellent semantic similarity performance on English text.

**Why RecursiveCharacterTextSplitter?**  
It respects Markdown heading structure (splitting at `## ` and `### ` boundaries first) before falling back to paragraph/line breaks, preserving semantic coherence within chunks.

**Anti-hallucination strategy:**  
The system prompt explicitly instructs the LLM to use *only* the retrieved context and to respond with the exact sentinel string `"Information not available in knowledge base."` if the context is insufficient. Temperature is set to `0.1` for maximum factual consistency.

---

## 📁 Project Structure

```
RAG-Based Company Knowledge Base Chatbot/
│
├── app.py                  # Streamlit UI application
├── rag_pipeline.py         # Core RAG pipeline (load → embed → retrieve → generate)
├── test_evaluation.py      # 22-question evaluation suite
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── .env                    # API keys (add your key here; not committed to git)
│
├── data/                   # Knowledge base documents
│   ├── company_profile.md
│   ├── products.md
│   ├── services.md
│   ├── pricing.md
│   ├── hr_policies.md
│   └── faqs.md
│
└── faiss_index/            # Auto-generated FAISS index (created on first run)
    ├── index.faiss
    └── index.pkl
```

---

## ⚙️ Setup Instructions

### Prerequisites
- Python 3.10 or higher
- An [OpenRouter](https://openrouter.ai/) account (free tier is sufficient)

### Step 1: Clone / Open the Project

```bash
cd "d:\RAG-Based Company Knowledge Base Chatbot"
```

### Step 2: Create a Virtual Environment (Recommended)

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ The first run will download the `all-MiniLM-L6-v2` model (~90 MB) and cache it locally. Subsequent runs will be instant.

### Step 4: Configure Your API Key

1. Sign up at [OpenRouter.ai](https://openrouter.ai/) (free)
2. Get your API key from the [Keys page](https://openrouter.ai/keys)
3. Open the `.env` file and replace the placeholder:

```env
OPENROUTER_API_KEY=sk-or-v1-your-actual-key-here
```

### Step 5: Run the Application

```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`.

On first run, the FAISS index will be built from the documents in `data/` and saved to `faiss_index/`. Subsequent runs will load the cached index in seconds.

---

## 🚀 How to Use the Chatbot

1. **Load KB** — The sidebar auto-loads the knowledge base on startup. Click **"▶ Load KB"** to reload or **"🔄 Rebuild"** to force re-index from the source documents.
2. **Ask Questions** — Type a question in the input box and press **"Ask →"**.
3. **View Citations** — Click **"📎 Sources"** under each answer to see which document chunks were used.
4. **Inspect Context** — Click **"🔍 Retrieved Context (raw)"** to see the exact text chunks passed to the LLM.
5. **Use Examples** — The sidebar lists example questions you can click to pre-fill the input.

---

## 🧪 Running the Evaluation Suite

```bash
python test_evaluation.py
```

This runs 22 questions categorized as:
| Category | Count | Tests |
|---|---|---|
| `direct_factual` | 8 | Single-document retrieval accuracy |
| `cross_document` | 5 | Multi-document reasoning |
| `paraphrased` | 5 | Semantic search robustness |
| `out_of_kb` | 4 | Hallucination prevention |

**Outputs generated:**
- `test_results.json` — machine-readable results with all details
- `test_report.txt` — human-readable summary report

**Pass/Fail logic (no hard-coded answers):**
- `out_of_kb` questions: PASS if the sentinel phrase is returned
- All other questions: PASS if a non-sentinel, non-empty answer is returned

---

## 🔧 Configuration Reference

All key settings are defined as constants at the top of [`rag_pipeline.py`](rag_pipeline.py):

| Parameter | Default | Description |
|---|---|---|
| `DATA_DIR` | `data/` | Directory containing `.md` knowledge base files |
| `FAISS_INDEX_PATH` | `faiss_index/` | Where the FAISS index is stored |
| `CHUNK_SIZE` | `800` | Characters per chunk |
| `CHUNK_OVERLAP` | `150` | Overlap between consecutive chunks |
| `TOP_K_RESULTS` | `5` | Number of chunks retrieved per query |
| `EMBEDDING_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace embedding model |
| `LLM_MODEL` | `meta-llama/llama-3.3-70b-instruct:free` | OpenRouter model ID |

---

## 📦 Key Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web UI framework |
| `langchain` + `langchain-community` | RAG orchestration, document loading, text splitting |
| `langchain-openai` | LLM client (ChatOpenAI) pointed at OpenRouter |
| `langchain-huggingface` | HuggingFace embedding integration |
| `faiss-cpu` | Fast vector similarity search |
| `sentence-transformers` | Downloads and runs `all-MiniLM-L6-v2` locally |
| `python-dotenv` | Loads `.env` file for API key management |
| `openai` | Required by `langchain-openai` as the HTTP client |

---

## 🛡️ Anti-Hallucination Guarantees

The chatbot enforces three layers of anti-hallucination protection:

1. **System Prompt Constraint** — The LLM is explicitly told it may only use the provided context and must return the exact sentinel phrase if context is insufficient.
2. **Low Temperature** — `temperature=0.1` minimizes creative generation and maximizes adherence to context.
3. **Sentinel Detection** — The UI checks the response for the sentinel phrase and renders a distinct warning UI element instead of a normal answer bubble.

---

## 🌐 Extending the Knowledge Base

To add new documents:
1. Drop any `.md` file into the `data/` directory.
2. Click **"🔄 Rebuild"** in the sidebar (or run `python rag_pipeline.py` from the CLI).
3. The FAISS index will be rebuilt to include the new content.

---

## 🚀 Streamlit Community Cloud Deployment (100% Free)

This repository is ready for 1-click deployment on **Streamlit Community Cloud**:
- **App URL:** Deploy via [share.streamlit.io](https://share.streamlit.io)
- **Secrets Setup:** Set `OPENROUTER_API_KEY` and `OPENROUTER_MODEL` in App Settings → Secrets.
- **Detailed Instructions:** See [DEPLOYMENT.md](DEPLOYMENT.md) for full instructions.

---

## 📜 License

This project is provided for educational and demonstration purposes. The company "Nexora Technologies" and all associated content are entirely fictional.