from django.urls import path
from .views import query_view, chatbot_page, departments_view, department_questions_view

urlpatterns = [
    path('query/', query_view, name='query'),
    path('chat/', chatbot_page, name='chatbot'),
    path('departments/', departments_view, name='departments'),
    path('department-questions/', department_questions_view, name='department_questions'),
] 