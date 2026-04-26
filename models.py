from django.db import models
import uuid
import os


def upload_path(instance, filename):
    """Generate unique upload path"""
    ext = filename.split('.')[-1]
    new_filename = f"{uuid.uuid4().hex}.{ext}"
    return os.path.join('uploads', new_filename)


class Dataset(models.Model):
    """Represents an uploaded file/dataset."""
    FILE_TYPE_CHOICES = [
        ('tabular', 'Tabular Data (CSV/Excel/Parquet/JSON)'),
        ('image', 'Image'),
        ('pdf', 'PDF Document'),
        ('text', 'Text File'),
        ('unknown', 'Unknown'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    original_filename = models.CharField(max_length=255)
    file = models.FileField(upload_to=upload_path)
    file_type = models.CharField(max_length=20, choices=FILE_TYPE_CHOICES, default='unknown')
    file_size = models.BigIntegerField(default=0)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    # Cached metadata
    rows = models.IntegerField(null=True, blank=True)
    columns = models.IntegerField(null=True, blank=True)
    column_info = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ['-uploaded_at']

    def __str__(self):
        return self.original_filename

    @property
    def file_size_human(self):
        size = self.file_size
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} TB"


class ChatMessage(models.Model):
    """Stores LLM chat history for a dataset."""
    ROLE_CHOICES = [('user', 'User'), ('assistant', 'Assistant')]

    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, related_name='messages')
    role = models.CharField(max_length=10, choices=ROLE_CHOICES)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['created_at']
