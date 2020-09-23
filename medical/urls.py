from django.urls import path
from . import views
from medspec import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', views.home, name='home'),
    path('download/', views.download_csv, name='download_csv'),
]
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)