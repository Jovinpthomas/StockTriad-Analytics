# stocks/forms.py
from django import forms
from .models import Stock
import yfinance as yf

class StockForm(forms.ModelForm):

    symbol = forms.CharField(label='Enter Stock Symbol:', required=True)

    class Meta:
        model = Stock
        fields = ['symbol', 'name', 'sector']
