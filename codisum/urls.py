from django.urls import path

from . import views

urlpatterns = [
    path('', views.generated, name='home'),
    path('about/', views.about, name='about'),
    path('api/', views.GenerateMsgs.as_view()),
]
