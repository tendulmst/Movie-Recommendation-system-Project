"""
URL Configuration for Movie Recommendation System
"""
from django.urls import path
from . import views

app_name = 'recommender'

urlpatterns = [
    # Main views
    path('', views.main, name='main'),
    path('fake-review/', views.fake_review_page, name='fake_review_page'),
    path('assistant/', views.chat_assistant_page, name='chat_assistant_page'),
    path('admin-dashboard/', views.admin_dashboard, name='admin_dashboard'),
    
    # API endpoints
    path('api/search/', views.search_movies, name='search_movies'),
    path('api/fake-review/', views.fake_review_api, name='fake_review_api'),
    path('api/watched/', views.watched_api, name='watched_api'),
    path('api/review/', views.review_api, name='review_api'),
    path('api/chat/', views.chat_assistant_api, name='chat_assistant_api'),
    path('api/chat/history/', views.chat_history_api, name='chat_history_api'),
    path('api/chat/clear/', views.chat_clear_api, name='chat_clear_api'),
   
     
    
    path('api/model-status/', views.model_status, name='model_status'),
    path('api/health/', views.health_check, name='health_check'),
]
