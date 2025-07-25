import sys
import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Add the project root (the directory containing 'qa') to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import QAPair after sys.path is set
import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'yaml_project.settings')
django.setup()
from qa.models import QAPair

# Parameters
MODEL_NAME = 'all-MiniLM-L6-v2'
PICKLE_PATH = 'qa/rag/chunks.pkl'
FAISS_INDEX_PATH = 'qa/rag/faiss.index'

# Load model
model = SentenceTransformer(MODEL_NAME)

# Load FAISS index
index = faiss.read_index(FAISS_INDEX_PATH)

# Load chunks and metadata
with open(PICKLE_PATH, 'rb') as f:
    data = pickle.load(f)
chunks = data['chunks']
meta = data['meta']

# Normalize embeddings for cosine similarity
faiss.normalize_L2(index.reconstruct_n(0, index.ntotal))


def retrieve(query, k=5):
    query_emb = model.encode([query]).astype('float32')
    faiss.normalize_L2(query_emb)
    D, I = index.search(query_emb, k)
    results = []
    seen_ids = set()
    for idx in I[0]:
        qa_id = meta[idx]['qa_id']
        if qa_id not in seen_ids:
            try:
                answer = QAPair.objects.get(id=qa_id).answer
                results.append(answer)
                seen_ids.add(qa_id)
            except QAPair.DoesNotExist:
                continue
        if len(results) >= k:
            break
    return results

QUESTION_INDEX_PATH = 'qa/rag/question_index.pkl'

def retrieve_best_question(user_query):
    if not os.path.exists(QUESTION_INDEX_PATH):
        return None, None, 0.0
    with open(QUESTION_INDEX_PATH, 'rb') as f:
        data = pickle.load(f)
    questions = data['questions']
    answers = data['answers']
    embeddings = data['embeddings']
    model = SentenceTransformer(MODEL_NAME)
    query_emb = model.encode([user_query]).astype('float32')
    norm_emb = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    norm_query = query_emb / np.linalg.norm(query_emb)
    sims = np.dot(norm_emb, norm_query[0])
    best_idx = int(np.argmax(sims))
    return questions[best_idx], answers[best_idx], float(sims[best_idx])


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Retrieve top-k relevant answers for a query.')
    parser.add_argument('query', type=str, help='User query')
    parser.add_argument('--k', type=int, default=5, help='Number of results to return')
    args = parser.parse_args()
    top_answers = retrieve(args.query, args.k)
    print('\n--- Top Answers ---')
    for i, answer in enumerate(top_answers, 1):
        print(f'{i}. {answer}\n') 