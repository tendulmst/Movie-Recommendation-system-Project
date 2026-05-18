from django.contrib import admin

from .models import ChatLog, ReviewCheckLog


@admin.register(ChatLog)
class ChatLogAdmin(admin.ModelAdmin):
    list_display = ("created_at", "role", "intent", "sentiment", "message_preview", "session_key")
    list_filter = ("role", "intent", "sentiment")
    search_fields = ("message", "session_key")
    readonly_fields = ("created_at",)

    @admin.display(description="Message")
    def message_preview(self, obj):
        return (obj.message or "")[:80]


@admin.register(ReviewCheckLog)
class ReviewCheckLogAdmin(admin.ModelAdmin):
    list_display = ("created_at", "movie_title", "label", "score", "sentiment")
    list_filter = ("label", "sentiment")
    search_fields = ("movie_title", "review_snippet")
