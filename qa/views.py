from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.views.decorators.http import require_GET
import json
from .models import QAPair
import difflib
import sys
import os
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle

# Small talk/greetings dataset
small_talk = {
    "hi": "Hello! How can I help you today?",
    "hello": "Hi there! Ask me anything about our services.",
    "hey": "Hey! How can I assist you?",
    "how are you": "I'm just a bot, but I'm here to help you!",
    "good morning": "Good morning! How can I help you today?",
    "good afternoon": "Good afternoon! How can I help you today?",
    "good evening": "Good evening! How can I help you today?",
    "thank you": "You're welcome! If you have more questions, just ask.",
    "thanks": "Happy to help! Let me know if you have more questions.",
    "what is your name": "I'm your YAML Q&A Bot!",
    "what is ur name": "I'm your YAML Q&A Bot!",
    "who are you": "I'm a helpful chatbot created to answer your questions.",
    "who made you": "I was created by my developer using Python and FastAPI!",
    "how old are you": "I'm as old as this project!",
    "what can you do": "I can answer your questions based on my knowledge base and have a friendly chat.",
    "where are you from": "I'm from the digital world, running on a server!",
    "what is your hobby": "Helping people and answering questions is my favorite thing!",
    "are you a robot": "Yes, I'm a virtual assistant bot.",
    "are you human": "No, I'm a bot, but I'm here to help you like a human would!",
    "do you have feelings": "I don't have feelings, but I care about helping you!",
    "can you help me": "Of course! Ask me anything.",
    "tell me a joke": "Why did the computer go to the doctor? Because it had a virus!"
}

# Add retriever import path
retriever_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'rag'))
if retriever_path not in sys.path:
    sys.path.insert(0, retriever_path)
import retriever

# Create your views here.

def format_as_points(answer):
    import re
    if '\n' in answer or len(answer) < 120:
        return answer
    sentences = re.split(r'(?<=[.!?]) +', answer)
    if len(sentences) <= 1:
        return answer
    points = [f'{i+1}. {s.strip()}' for i, s in enumerate(sentences) if s.strip()]
    return '\n'.join(points)

def semantic_question_search(user_query, threshold=0.6):
    # Load question embeddings and metadata
    QUESTION_INDEX_PATH = 'qa/rag/question_index.pkl'
    if not os.path.exists(QUESTION_INDEX_PATH):
        return None, None
    with open(QUESTION_INDEX_PATH, 'rb') as f:
        data = pickle.load(f)
    questions = data['questions']
    answers = data['answers']
    embeddings = data['embeddings']
    # Embed user query
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_emb = model.encode([user_query]).astype('float32')
    # Compute cosine similarity
    norm_emb = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    norm_query = query_emb / np.linalg.norm(query_emb)
    sims = np.dot(norm_emb, norm_query[0])
    best_idx = int(np.argmax(sims))
    if sims[best_idx] >= threshold:
        return questions[best_idx], answers[best_idx]
    return None, None

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def find_best_yaml_match(user_query, qa_data, threshold=0.75):
    # Try exact match first
    for qa in qa_data:
        if user_query.strip().lower() == qa['question'].strip().lower():
            return qa['answer'], 1.0, qa['question']
    # Fuzzy match using difflib
    questions = [qa['question'] for qa in qa_data]
    matches = difflib.get_close_matches(user_query, questions, n=1, cutoff=0.0)
    if matches:
        best_q = matches[0]
        score = difflib.SequenceMatcher(None, user_query, best_q).ratio()
        answer = next(qa['answer'] for qa in qa_data if qa['question'] == best_q)
        return answer, score, best_q
    return None, 0.0, None

@csrf_exempt
@require_POST
def query_view(request):
    data = json.loads(request.body)
    user_query = data.get('query', '').strip()
    is_dropdown = data.get('isDropdown', False)
    if not user_query:
        return JsonResponse({'error': 'No query provided.'}, status=400)

    # 0. Small talk/greetings check (case-insensitive, fuzzy)
    user_query_lower = user_query.lower()
    st_keys = list(small_talk.keys())
    st_match = difflib.get_close_matches(user_query_lower, st_keys, n=1, cutoff=0.85)
    if st_match:
        return JsonResponse({'answer': small_talk[st_match[0]], 'source': 'small_talk'})

    # Load all QAPair as list of dicts for YAML/DB match
    qa_data = list(QAPair.objects.values('question', 'answer', 'department'))

    # 1. YAML/DB match (exact/fuzzy)
    answer, score, matched_q = find_best_yaml_match(user_query, qa_data, threshold=0.75)
    if answer and score >= 0.75:
        return JsonResponse({'answer': format_as_points(answer), 'source': 'yaml', 'confidence': score, 'matched_question': matched_q})

    # 2. RAG fallback: semantic search for closest question, return its answer
    import sys
    retriever_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'rag'))
    if retriever_path not in sys.path:
        sys.path.insert(0, retriever_path)
    import retriever
    matched_q, best_answer, sim_score = retriever.retrieve_best_question(user_query)
    return JsonResponse({'answer': format_as_points(best_answer), 'source': 'rag', 'confidence': sim_score, 'matched_question': matched_q})

@require_GET
def departments_view(request):
    departments = list(QAPair.objects.exclude(department='').values_list('department', flat=True).distinct())
    return JsonResponse({'departments': departments})

@require_GET
def department_questions_view(request):
    dept = request.GET.get('department', '')
    start = int(request.GET.get('start', 0))
    limit = int(request.GET.get('limit', 5))
    # Get all QAPair objects for the department
    qas = QAPair.objects.filter(department__iexact=dept)
    # Only keep the first question for each unique (answer, department) pair
    seen = set()
    unique_questions = []
    for qa in qas:
        key = (qa.answer.strip().lower(), qa.department.strip().lower())
        if key not in seen:
            seen.add(key)
            unique_questions.append(qa.question)
    paginated = unique_questions[start:start+limit]
    has_more = start + limit < len(unique_questions)
    return JsonResponse({'questions': paginated, 'has_more': has_more})

def chatbot_page(request):
    return render(request, 'qa/chatbot.html')
