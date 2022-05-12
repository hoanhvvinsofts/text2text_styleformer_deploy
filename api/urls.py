# from ... import urls_partenr ... -> re_path, path .. 
from django.urls import path
from .views import ServiceText2Text

app_name = "api"
urlpatterns=[
    path("text2text", ServiceText2Text.as_view()),
]