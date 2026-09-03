# 🚀 Streamlit Community Cloud Deployment Guide (100% Free)

This guide walks you through deploying the **Nexora Technologies RAG Knowledge Base Chatbot** to **Streamlit Community Cloud** completely free of charge.

---

## 💡 Why This Setup Is 100% Free Forever

| Component | Provider / Tech | Free Tier Details |
| :--- | :--- | :--- |
| **App Hosting** | **Streamlit Community Cloud** | 100% Free for public GitHub repos (1 GB RAM, 1 CPU, HTTPS included) |
| **LLM Inference** | **OpenRouter Free Tier** | Free access to models like `nvidia/nemotron-3.5-lightning:free` |
| **Embeddings** | **HuggingFace (`all-MiniLM-L6-v2`)** | Runs locally on CPU via `sentence-transformers` ($0 cost, no API required) |
| **Vector Database** | **FAISS (CPU)** | Local vector search running inside the container ($0 cost, no cloud DB fees) |

---

## 📋 Prerequisites

1. A **GitHub** account.
2. A **Streamlit Community Cloud** account (Sign in with your GitHub at [share.streamlit.io](https://share.streamlit.io)).
3. An **OpenRouter API Key** (Generate a free key at [openrouter.ai/keys](https://openrouter.ai/keys)).

---

## 🛠️ Step-by-Step Deployment Instructions

### Step 1: Push Your Code to GitHub

Make sure all your latest changes are pushed to your GitHub repository:

```bash
git add .
git commit -m "Configure project for Streamlit Community Cloud deployment"
git push origin main
```

*(Note: `.env` and `faiss_index/` are automatically excluded by `.gitignore` to keep your credentials safe. The app will automatically build the FAISS index from `data/` on its first launch).*

---

### Step 2: Create a New App on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) and log in with your GitHub account.
2. Click the **"New app"** (or **"Create app"**) button.
3. Select your repository:
   - **Repository:** `YourGitHubUsername/RAG-Based-Company-Knowledge-Base-Chatbot-`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Click **"Advanced settings..."** *before* deploying (or configure Secrets immediately after).

---

### Step 3: Configure Streamlit Secrets (Crucial!)

In the **Secrets** section of the Streamlit Cloud app setup (or under **App Settings → Secrets** after creation), paste the following:

```toml
# OpenRouter API Key (Free from https://openrouter.ai/keys)
OPENROUTER_API_KEY = "sk-or-v1-your-actual-openrouter-key-here"

# 100% Free Model on OpenRouter
OPENROUTER_MODEL = "nvidia/nemotron-3.5-lightning:free"
```

> **Note:** The app automatically reads from `st.secrets` when deployed on Streamlit Cloud, and falls back to `.env` when running locally on your computer.

---

### Step 4: Click Deploy! 🚀

1. Click **"Deploy!"**.
2. Streamlit Cloud will:
   - Spin up a fresh Linux container.
   - Install all packages from `requirements.txt`.
   - Download the compact `all-MiniLM-L6-v2` embedding model (~80MB).
   - Automatically parse the company documents in `data/` and build the local FAISS index.
3. Within 1–2 minutes, your enterprise chatbot will be live with a public URL (e.g. `https://your-app-name.streamlit.app`)!

---

## ⚙️ Maintenance & Updates

- **Updating Documents:** To add new corporate documents, simply commit them to the `data/` folder and push to `main`. Streamlit Cloud will automatically detect the push and re-index.
- **Uploading Files in Live App:** Users can also drag-and-drop `.pdf`, `.docx`, `.txt`, and `.md` files directly in the sidebar; the app dynamically chunks and indexes them in real time.
- **Changing the Model:** You can switch to any other free model (e.g., `google/gemma-4-31b-it:free`, `minimax/minimax-m2.7:free`) simply by editing `OPENROUTER_MODEL` in your Streamlit Cloud **Secrets** dashboard without redeploying code.
