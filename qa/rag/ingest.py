import sys
import os

# Add the project root (the directory containing 'qa') to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print("sys.path:", sys.path)  # Debug: see what paths are set

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'yaml_project.settings')
import django
django.setup()

import pickle
import faiss
from sentence_transformers import SentenceTransformer
from qa.models import QAPair
import numpy as np
from qa.rag.ingest_logic import run_ingest

if __name__ == '__main__':
    run_ingest() 