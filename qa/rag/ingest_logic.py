import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from qa.models import QAPair
import django
import numpy as np

# Setup Django environment if running as a script
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'yaml_project.settings')
django.setup()

MODEL_NAME = 'all-MiniLM-L6-v2'
PICKLE_PATH = 'qa/rag/chunks.pkl'
FAISS_INDEX_PATH = 'qa/rag/faiss.index'
CHUNK_SIZE = 256  # characters
QUESTION_INDEX_PATH = 'qa/rag/question_index.pkl'

def run_ingest():
    model = SentenceTransformer(MODEL_NAME)
    qas = QAPair.objects.all()
    chunks = []
    meta = []
    # For question semantic search
    question_texts = []
    question_answers = []
    question_departments = []
    for qa in qas:
        # For chunked answer retrieval
        answer = qa.answer
        for i in range(0, len(answer), CHUNK_SIZE):
            chunk = answer[i:i+CHUNK_SIZE]
            chunks.append(chunk)
            meta.append({'qa_id': qa.id, 'start': i, 'end': i+CHUNK_SIZE})
        # For question semantic search
        question_texts.append(qa.question)
        question_answers.append(qa.answer)
        question_departments.append(qa.department)
    # Embeddings for chunks
    embeddings = model.encode(chunks, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32')
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(PICKLE_PATH, 'wb') as f:
        pickle.dump({'chunks': chunks, 'meta': meta}, f)
    # Embeddings for questions
    question_embeddings = model.encode(question_texts, show_progress_bar=True)
    question_embeddings = np.array(question_embeddings).astype('float32')
    with open(QUESTION_INDEX_PATH, 'wb') as f:
        pickle.dump({
            'questions': question_texts,
            'answers': question_answers,
            'departments': question_departments,
            'embeddings': question_embeddings
        }, f)
    print(f"Saved {len(chunks)} chunks, FAISS index, and pickle file.")
    print(f"Saved {len(question_texts)} question embeddings for semantic search.") 