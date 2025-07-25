from django.db import models

# Create your models here.

class QAPair(models.Model):
    question = models.TextField()
    answer = models.TextField()
    department = models.CharField(max_length=100, blank=True)

    def __str__(self):
        return f"{self.question[:50]}..."

# --- Auto-ingest on save/delete ---
from django.db.models.signals import post_save, post_delete
from django.dispatch import receiver

def _run_ingest_async():
    import threading
    def task():
        from qa.rag.ingest_logic import run_ingest
        run_ingest()
    threading.Thread(target=task).start()

@receiver(post_save, sender=QAPair)
def reingest_on_save(sender, **kwargs):
    _run_ingest_async()

@receiver(post_delete, sender=QAPair)
def reingest_on_delete(sender, **kwargs):
    _run_ingest_async()
