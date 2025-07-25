from django.contrib import admin
from .models import QAPair

@admin.register(QAPair)
class QAPairAdmin(admin.ModelAdmin):
    list_display = ('question', 'answer', 'department')
    search_fields = ('question', 'answer', 'department')
    list_filter = ('department',)  # Add department filter in admin
