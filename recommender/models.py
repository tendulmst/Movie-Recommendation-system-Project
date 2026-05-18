from django.db import models


class ChatLog(models.Model):
    session_key = models.CharField(max_length=64, db_index=True)
    role = models.CharField(max_length=16)
    message = models.TextField()
    intent = models.CharField(max_length=48, blank=True, default="")
    sentiment = models.CharField(max_length=16, blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.role}: {self.message[:40]}"


class ReviewCheckLog(models.Model):
    movie_title = models.CharField(max_length=255, blank=True, default="")
    review_snippet = models.TextField(blank=True, default="")
    label = models.CharField(max_length=16)
    score = models.FloatField(default=0.0)
    sentiment = models.CharField(max_length=16, blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.movie_title or 'review'} — {self.label}"
