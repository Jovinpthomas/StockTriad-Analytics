# stocks/models.py
from django.db import models

class Stock(models.Model):
    symbol = models.CharField(max_length=10, blank=True, null=True)
    name = models.CharField(max_length=255, blank=True, null=True)
    sector = models.CharField(max_length=255, blank=True, null=True)

    def __str__(self):
        return f"{self.symbol} - {self.name}"
