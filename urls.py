from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('upload/', views.upload_file, name='upload_file'),
    path('dataset/<uuid:dataset_id>/', views.dataset_view, name='dataset_view'),
    path('dataset/<uuid:dataset_id>/delete/', views.delete_dataset, name='delete_dataset'),
    path('dataset/<uuid:dataset_id>/chart/', views.generate_chart, name='generate_chart'),
    path('dataset/<uuid:dataset_id>/clean/', views.clean_dataset, name='clean_dataset'),
    path('dataset/<uuid:dataset_id>/chat/', views.chat_message, name='chat_message'),
    path('dataset/<uuid:dataset_id>/chat/clear/', views.clear_chat, name='clear_chat'),
    path('dataset/<uuid:dataset_id>/export-pdf/', views.export_pdf, name='export_pdf'),
    path('dataset/<uuid:dataset_id>/download-csv/', views.download_data, name='download_data'),
]
