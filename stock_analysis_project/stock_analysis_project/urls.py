# stock_analysis_project/urls.py
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('stocks.urls')),  # Include your stocks app URLs here
]
