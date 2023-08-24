from django.urls import path

from .views import Index, Success_View, Whatsapp_Chat, Whatsapp_Use_View

urlpatterns = [
    path("", Index.as_view()),
    path(
        "wha/",
        Whatsapp_Chat.as_view(),
    ),
    path("wha/success/", Success_View.as_view()),
    path("whatsapp/", Whatsapp_Use_View.as_view()),
]
