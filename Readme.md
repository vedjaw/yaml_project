# YAML Bot - Django Version (instructions.md)

## 📌 Project Overview
This project implements a chatbot that answers predefined questions from a YAML file. It supports normal retrieval (exact and semantic) and allows for conditional follow-ups (e.g., claim type).

## 🛠 Tech Stack
- Django (backend framework)
- Django REST Framework (API layer)
- Sentence Transformers (semantic search)
- PyYAML (YAML parsing)
- SQLite / PostgreSQL (DB)
- FAISS (optional for RAG)
- OpenAI or HuggingFace (optional for response generation)

## 🚀 Core Functionalities
- Load Q&A pairs from `qa_dataset.yaml`
- Match user query to best answer
- Follow-up logic for multi-step responses
- API endpoint `/query`
- Optional RAG integration (see instructions_rag.md)

## 📂 Project Structure
```
yaml_project/
├── qa/
│   ├── models.py
│   ├── views.py
│   ├── urls.py
│   ├── management/
│   │   └── commands/
│   │       └── import_qa_yaml.py
│   ├── rag/
│   │   ├── ingest.py
│   │   ├── retriever.py
│   │   └── generator.py
│   ├── services/
│       └── query_engine.py
├── qa_dataset.yaml
├── manage.py
└── requirements.txt
```

## 🧾 Instructions

### Step 1: Create Project & App
```bash
django-admin startproject yaml_project
cd yaml_project
python manage.py startapp qa
```

### Step 2: Define QAPair model
In `qa/models.py`:
```python
from django.db import models

class QAPair(models.Model):
    question = models.TextField()
    answer = models.TextField()
    department = models.CharField(max_length=100, blank=True)
```

### Step 3: Register app
In `settings.py`:
```python
INSTALLED_APPS = [
    ...
    'qa',
]
```

### Step 4: Import YAML
Create `qa/management/commands/import_qa_yaml.py`:
```python
import os, yaml
from pathlib import Path
from django.core.management.base import BaseCommand
from qa.models import QAPair

class Command(BaseCommand):
    help = 'Import Q&A pairs from qa_dataset.yaml'

    def handle(self, *args, **kwargs):
        yaml_path = os.path.join(Path.home(), "Desktop/DjangoProjects/yaml bot/data/qa_dataset.yaml")
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        for entry in data:
            dept = entry.get('department', '')
            questions = entry.get('questions', [entry.get('question')])
            answer = entry.get('answer', '')
            for q in questions:
                QAPair.objects.get_or_create(department=dept, question=q, defaults={'answer': answer})
        self.stdout.write(self.style.SUCCESS('Imported Q&A pairs from YAML.'))
```

### Step 5: Build Query Endpoint
Use DRF to create `/query/` route that:
- Takes `{"query": "..."}` as input
- Returns matched answer or follow-up options
- Uses sentence transformer for semantic similarity

Use helper functions from `query_engine.py` under `qa/services/`.

### Step 6: (Optional) Add Frontend
Use Django templates or basic HTML to make a form with a chatbot interface.
