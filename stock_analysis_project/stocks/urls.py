# stocks/urls.py
from django.urls import path
from .views import index, profile, charts, news, insights, analysts, risks, financials_balance_sheet, financials_income_statement, financials_cash_flow, holders, peers, indices, clear_session

urlpatterns = [
    path('', index, name='index'),
    path('profile/', profile, name='profile'),
    path('charts/', charts, name='charts'),
    path('news/', news, name='news'),
    path('insights/', insights, name='insights'),
    path('analysts/', analysts, name='analysts'),
    path('risks/', risks, name='risks'),
    path('financials-balance-sheet/', financials_balance_sheet, name='financials_balance_sheet'),
    path('financials-income-statement/', financials_income_statement, name='financials_income_statement'),
    path('financials-cash-flow/', financials_cash_flow, name='financials_cash_flow'),
    path('holders/', holders, name='holders'),
    path('peers/', peers, name='peers'),
    path('indices/', indices, name='indices'),
    path('clear-session/', clear_session, name='clear_session')
]
