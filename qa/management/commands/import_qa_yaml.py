import os
import yaml
from django.core.management.base import BaseCommand
from qa.models import QAPair

class Command(BaseCommand):
    help = 'Import Q&A pairs from data/qa_dataset.yaml'

    def handle(self, *args, **kwargs):
        yaml_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'data', 'qa_dataset.yaml')
        if not os.path.exists(yaml_path):
            self.stdout.write(self.style.ERROR(f'YAML file not found at {yaml_path}'))
            return
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        count = 0
        for entry in data:
            dept = entry.get('department', '')
            questions = entry.get('questions') or [entry.get('question')]
            answer = entry.get('answer', '')
            if not questions or not answer:
                continue
            for q in questions:
                obj, created = QAPair.objects.get_or_create(
                    question=q,
                    department=dept,
                    defaults={'answer': answer}
                )
                if created:
                    count += 1
        self.stdout.write(self.style.SUCCESS(f'Imported {count} Q&A pairs from YAML.')) 