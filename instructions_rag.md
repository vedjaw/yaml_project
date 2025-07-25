# YAML Bot with RAG - Django Version (instructions_rag.md)

## 📌 Overview
This extends the Django YAML bot by integrating Retrieval Augmented Generation (RAG) using FAISS and Sentence Transformers.

## 🛠 Tech Stack
- Django
- Sentence Transformers
- FAISS
- PyYAML
- Pickle
- OpenAI (optional for completion)

## 📁 Files
- `rag/ingest.py` → Builds vector index from `qa_dataset.yaml`
- `rag/retriever.py` → Retrieves top-k semantically matched chunks
- `rag/generator.py` → (Optional) Generates refined answers using GPT

## ⚙️ How It Works
1. `qa_dataset.yaml` is split into small chunks
2. Each chunk is embedded and indexed into FAISS
3. Query is embedded and searched in FAISS
4. Retrieved context is returned (or passed to LLM)

## 📂 Directory Structure
```
qa/
├── rag/
│   ├── ingest.py
│   ├── retriever.py
│   └── generator.py
```

## 🧾 Instructions

### Step 1: Install Dependencies
```bash
pip install faiss-cpu sentence-transformers
```

### Step 2: Ingest Documents
In `qa/rag/ingest.py`:
- Read `qa_dataset.yaml`
- Split answers into 500-char chunks
- Embed and store them in `rag/vector.index`
```bash
python qa/rag/ingest.py
```

### Step 3: Retrieve Matches
In `qa/rag/retriever.py`, create:
```python
from sentence_transformers import SentenceTransformer
import faiss, pickle, os

EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
VECTOR_DB_PATH = 'rag/vector.index'

embedder = SentenceTransformer(EMBEDDING_MODEL)

with open(VECTOR_DB_PATH + '.pkl', 'rb') as f:
    docs = pickle.load(f)
index = faiss.read_index(VECTOR_DB_PATH)

def retrieve(query, k=3):
    query_emb = embedder.encode([query])
    D, I = index.search(query_emb, k)
    return [docs[i] for i in I[0]]
```

### Step 4: Connect to Query Flow
From `query_engine.py`, import `retrieve(query)` and use its output when answering ambiguous queries.

### Step 5: (Optional) GPT Generation
In `generator.py`, call OpenAI with context:
```python
openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[{"role": "system", "content": context}, {"role": "user", "content": query}]
)
```

## ✅ Done!
Now your Django chatbot has both:
- Direct YAML Q&A matching
- Semantic retrieval from chunked YAML answers