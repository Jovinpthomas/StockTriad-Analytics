# stocks/views.py
from django.shortcuts import render, redirect
from .forms import StockForm
from django.http import HttpResponseServerError
from django.http import HttpResponseBadRequest
from .models import Stock
import pandas as pd
import yfinance as yf
import datetime
import plotly.io as pio
import plotly.graph_objects as go
import requests
from yahoo_fin.stock_info import get_holders
from prettytable import PrettyTable
from collections import deque
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from string import punctuation
from newspaper import Article
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

ALPHA_VANTAGE_API_KEY = 'your_alpha_vantage_api_key'
POLYGON_API_KEY = 'your_polygon_api_key'

def extract_date(date_string):
    date_only = date_string.date() if isinstance(date_string, pd.Timestamp) else date_string
    return date_only.strftime("%d/%m/%Y")

def format_large_number(number):
    try:
        parts = str(number).split('.')
        parts[0] = "{:,}".format(int(parts[0]))
        return '.'.join(parts)
    except (ValueError, TypeError):
        return str(number)
    
def get_ticker_name(ticker_symbol):
    try:
        ticker = yf.Ticker(ticker_symbol)
        
        long_name = ticker.info.get('longName')
        if long_name:
            return long_name
        
        short_name = ticker.info.get('shortName')
        if short_name:
            return short_name
                
        return ticker_symbol
    except Exception as e:
        print(f"Error occurred while fetching ticker name for {ticker_symbol}: {e}")
        return None

def get_top_n_peers(symbol, n):
    peers_info = []
    stock = yf.Ticker(symbol)
    
    news_data1 = fetch_news(symbol, POLYGON_API_KEY)
    tickers_list = extract_tickers(news_data1)
    
    news_data2 = stock.news
    related_tickers = extract_related_tickers(news_data2)
    
    combined_tickers = list(set(related_tickers + tickers_list))
    if symbol in combined_tickers:
        combined_tickers.remove(symbol)
    
    for peer_symbol in combined_tickers:
        try:
            peer_data = yf.Ticker(peer_symbol)
            if 'marketCap' in peer_data.info and peer_data.info['quoteType'] == 'EQUITY' and (stock.info['industry'] == peer_data.info['industry'] or stock.info['sector'] == peer_data.info['sector']):
                market_cap = peer_data.info['marketCap']
                peers_info.append({'Symbol': peer_symbol, 'MarketCap': market_cap})
        except Exception as e:
            print(f"Error processing peer symbol {peer_symbol}: {e}")

    peers_df = pd.DataFrame(peers_info)

    if not peers_df.empty:
        top_n_peers = peers_df.nlargest(n, 'MarketCap')['Symbol'].tolist()
        return top_n_peers
    else:
        return []
    
def get_top_n_indices(stock_symbols, n, selected_period_value):
    stocks_info = []
    for symbol in stock_symbols:
        try:
            stock_data = yf.Ticker(symbol)
            if 'marketCap' in stock_data.info:
                market_cap = stock_data.info['marketCap']
                stocks_info.append({'Symbol': symbol, 'MarketCap': market_cap})
            elif stock_data.info['quoteType'] == 'INDEX':
                index_data = yf.download(symbol, period = selected_period_value)
                if not index_data.empty:
                    index_market_cap = index_data['Adj Close'].iloc[-1] * index_data['Volume'].iloc[-1]
                    stocks_info.append({'Symbol': symbol, 'MarketCap': index_market_cap})
        except Exception as e:
            print(f"Error processing symbol {symbol}: {e}")

    stocks_df = pd.DataFrame(stocks_info)

    if not stocks_df.empty:
        top_n_indices = stocks_df.nlargest(n, 'MarketCap')['Symbol'].tolist()
        return top_n_indices
    else:
        return []

def generate_stock_price_chart(symbol, selected_time_period, selected_slider_value):
    if selected_time_period == '1d':
        data = yf.download(tickers=symbol, period=selected_time_period, interval='60m')
    else:
        data = yf.download(tickers=symbol, period=selected_time_period)
        
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close', 
                             hoverinfo='x+y+text', 
                             text=['Open: {:.2f}<br>High: {:.2f}<br>Low: {:.2f}<br>Close: {:.2f}<br>Volume: {:.0f}'.format(open_val, high_val, low_val, close_val, volume_val) 
                                   for open_val, high_val, low_val, close_val, volume_val in zip(data['Open'], data['High'], data['Low'], data['Close'], data['Volume'])]))

    fig.update_layout(
        title=f"{symbol} Stock Price Chart for {selected_slider_value}",
        xaxis_title="Time",
        yaxis_title="Price",
        height=400,
        width=800,
    )

    plot_html = pio.to_html(fig, full_html=False)
    return plot_html

def plot_stock_peer_price_change_percentage(stock_tickers, selected_time_period, selected_slider_value):
    fig = go.Figure()
    
    for stock_ticker in stock_tickers:
        if selected_time_period == '1d':
            stock_data = yf.download(tickers=stock_ticker, period=selected_time_period, interval='60m')
        else:
            stock_data = yf.download(tickers=stock_ticker, period=selected_time_period)
        
        stock_data['Close_Percentage_Change'] = stock_data['Close'].pct_change() * 100
        
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close_Percentage_Change'], mode='lines', name=get_ticker_name(stock_ticker)))
    
    fig.update_layout(
        title=f"Percentage Change in Closing Price for {selected_slider_value}",
        xaxis_title="Time",
        yaxis_title="Percentage Change",
        height=400,
        width=800,
    )

    plot_html = pio.to_html(fig, full_html=False)
    return plot_html

def plot_stock_index_price_change_percentage(stock_tickers,  selected_time_period, selected_slider_value):
    fig = go.Figure()
    
    for stock_ticker in stock_tickers:
        if selected_time_period == '1d':
            data = yf.download(tickers=stock_ticker, period=selected_time_period, interval='60m')
        else:
            data = yf.download(tickers=stock_ticker, period=selected_time_period)
        
        data['Close_Percentage_Change'] = data['Close'].pct_change() * 100
        
        fig.add_trace(go.Scatter(x=data.index, y=data['Close_Percentage_Change'], mode='lines', name=get_ticker_name(stock_ticker)))
    
    fig.update_layout(
        title=f"Percentage Change in Closing Price for {selected_slider_value}",
        xaxis_title="Time",
        yaxis_title="Percentage Change",
        height=400,
        width=800,
    )

    plot_html = pio.to_html(fig, full_html=False)
    return plot_html

def fetch_data(symbol, api_key, selected_time_period, outputsize='full'):
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_{selected_time_period.upper()}&symbol={symbol}&apikey={api_key}&outputsize={outputsize}'
    response = requests.get(url)
    data = response.json()
    return data

def calculate_custom_sma(data, window, time_series_key):
    try:
        if time_series_key not in data:
            raise KeyError(f"Key '{time_series_key}' not found in data dictionary.")
        
        df = pd.DataFrame(data[time_series_key]).T
        df.index = pd.to_datetime(df.index)
        
        # Check if the required columns are present in the DataFrame
        if '4. close' not in df.columns:
            raise KeyError("Column '4. close' not found in the DataFrame.")

        df['Close'] = df['4. close'].astype(float)
        
        sma = df['Close'].rolling(window=window).mean()
        return sma
    
    except KeyError as e:
        print(f"Error: {e}")
        return None
    except ValueError as e:
        print(f"Error converting data to float: {e}")
        return None
    
def plot_prices_with_sma(symbol, data, sma_50, sma_200, selected_slider_value, time_series_key):
    df = pd.DataFrame(data[time_series_key]).T
    df.index = pd.to_datetime(df.index)
    df['Close'] = df['4. close'].astype(float)

    trace_price = go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Price')
    trace_sma_50 = go.Scatter(x=df.index, y=sma_50, mode='lines', name='50-day SMA')
    trace_sma_200 = go.Scatter(x=df.index, y=sma_200, mode='lines', name='200-day SMA')

    layout = go.Layout(title=f'{symbol} Historical Prices with SMA ({selected_slider_value})', xaxis=dict(title='Date'), yaxis=dict(title='Price'))
    fig = go.Figure(data=[trace_price, trace_sma_50, trace_sma_200], layout=layout)
    
    plot_html = pio.to_html(fig, full_html=False)
    return plot_html

def plot_bollinger_bands(symbol, api_key, selected_time_period, selected_slider_value, time_period=20, series_type='close'):
    endpoint = f'https://www.alphavantage.co/query?function=BBANDS&symbol={symbol}&interval={selected_time_period}&time_period={time_period}&series_type={series_type}&apikey={api_key}'

    try:
        response = requests.get(endpoint)
        data = response.json()
        
        dates = list(data['Technical Analysis: BBANDS'].keys())
        upper_band = [float(data['Technical Analysis: BBANDS'][date]['Real Upper Band']) for date in dates]
        middle_band = [float(data['Technical Analysis: BBANDS'][date]['Real Middle Band']) for date in dates]
        lower_band = [float(data['Technical Analysis: BBANDS'][date]['Real Lower Band']) for date in dates]

        trace_upper = go.Scatter(x=dates, y=upper_band, mode='lines', name='Upper Band')
        trace_middle = go.Scatter(x=dates, y=middle_band, mode='lines', name='Middle Band')
        trace_lower = go.Scatter(x=dates, y=lower_band, mode='lines', name='Lower Band')

        layout = go.Layout(title=f'{symbol} Bollinger Bands ({selected_slider_value})', xaxis=dict(title='Date'), yaxis=dict(title='Price'))

        fig = go.Figure(data=[trace_upper, trace_middle, trace_lower], layout=layout)
        plot_html = pio.to_html(fig, full_html=False)

        return plot_html

    except Exception as e:
        print(f"Error occurred: {e}")

def get_rsi_data(symbol, api_key, selected_time_period, time_period=14):
    url = f"https://api.polygon.io/v1/indicators/rsi/{symbol}?timespan={selected_time_period}&window={time_period}&series_type=close&order=desc&apiKey={api_key}"
    response = requests.get(url)
    data = response.json()
    
    if 'results' in data and 'values' in data['results']:
        rsi_values = data['results']['values']
        timestamps = [value['timestamp'] for value in rsi_values]
        rsi = [value['value'] for value in rsi_values]
        
        df = pd.DataFrame({'timestamp': timestamps, 'rsi': rsi})
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    else:
        print("No RSI data found in the response.")
        return None

def compute_rsi(data, time_period=14):
    delta = data.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(time_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(time_period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def plot_rsi(symbol, df, selected_slider_value):
    if df is not None and not df.empty: 
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=df.index, y=df['rsi'], mode='lines', name='RSI'))
        fig.update_layout(
            title=f'{symbol} Relative Strength Index ({selected_slider_value})',
            xaxis_title='Date',
            yaxis_title='RSI',
            height=400,
            width=800,
        )

        plot_html = pio.to_html(fig, full_html=False)
        return plot_html
    else:
        return None

def fetch_news(choice, api_key):
    url = f"https://api.polygon.io/v2/reference/news?ticker={choice}&apiKey={api_key}"
    response = requests.get(url)
    
    if response.status_code == 200:
        return response.json()
    else:
        print("Failed to fetch news:", response.text)
        return None

def read_article(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        print("Error occurred:", str(e))
        return None

def preprocess_text(text):
    text = text.lower()

    text = ''.join([c for c in text if c not in punctuation])

    tokens = word_tokenize(text)

    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]

    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]

    preprocessed_text = ' '.join(lemmatized_tokens)
    return preprocessed_text
    
def get_sentiment(text):
    blob = TextBlob(text)
    sentiment_score = "{:.3f}".format(blob.sentiment.polarity)
    sentiment_score = float(sentiment_score)
    if sentiment_score > 0:
        sentiment = 'positive'
    elif sentiment_score < 0:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'
    return sentiment_score, sentiment
  
def extract_tickers(news_data):
    tickers_list = []

    if news_data:
        for article in news_data['results']:
            tickers_list.extend(article['tickers'])
        
    return tickers_list

def extract_related_tickers(news_data):
    related_tickers = []

    if news_data:
        for article in news_data:
            related_tickers.extend(article.get('relatedTickers', []))

    return related_tickers

def get_peer_table(choice, combined_tickers):
    def get_ticker_info(ticker):
        try:
            ticker_info = yf.Ticker(ticker)
            
            return {
                'Ticker': ticker,
                'Name': ticker_info.info.get('longName'),
                'Sector': ticker_info.info.get('sector'),
                'Industry': ticker_info.info.get('industry'),
                'Market Capitalization': format_large_number("{:.2f}".format(ticker_info.info.get('marketCap'))) if ticker_info.info.get('marketCap') is not None else None,
            }
        except Exception as e:
            print(f"Error occurred while fetching data for {ticker}: {e}")
            return None

    table = PrettyTable()
    table.field_names = ['Ticker', 'Name', 'Sector', 'Industry', 'Market Capitalization']
    
    choice_info = get_ticker_info(choice)
    if choice_info: 
        table.add_row([choice_info['Ticker'], choice_info['Name'], choice_info['Sector'], choice_info['Industry'], choice_info['Market Capitalization']])
        
    for ticker in combined_tickers:
        ticker_info = get_ticker_info(ticker)
        if ticker_info:
            table.add_row([ticker_info['Ticker'], ticker_info['Name'], ticker_info['Sector'], ticker_info['Industry'], ticker_info['Market Capitalization']])
    
    return table

def get_valuation_metrics_table(choice, tickers_list):
    def get_valuation_metrics(ticker):
        try:
            ticker_info = yf.Ticker(ticker)
            
            return {
                'Ticker': ticker,
                'Trailing P/E Ratio': "{:.2f}".format(ticker_info.info.get('trailingPE')) if ticker_info.info.get('trailingPE') is not None else None,
                'P/S Ratio': "{:.2f}".format(ticker_info.info.get('priceToSalesTrailing12Months')) if ticker_info.info.get('priceToSalesTrailing12Months') is not None else None,
                'EV/EBITDA Ratio': "{:.2f}".format(ticker_info.info.get('enterpriseToEbitda')) if ticker_info.info.get('enterpriseToEbitda') is not None else None,
                'Trailing PEG Ratio': "{:.2f}".format(ticker_info.info.get('trailingPegRatio')) if ticker_info.info.get('trailingPegRatio') is not None else None,
                'Dividend Yield': "{:.2f}".format(ticker_info.info.get('dividendYield')) if ticker_info.info.get('dividendYield') is not None else None,
            }

        except Exception as e:
            print(f"Error occurred while fetching data for {ticker}: {e}")
            return None
        
    table = PrettyTable()
    table.field_names = ['Ticker', 'Trailing P/E Ratio', 'P/S Ratio', 'EV/EBITDA Ratio', 'Trailing PEG Ratio', 'Dividend Yield (%)']
    
    choice_metrics = get_valuation_metrics(choice)
    if choice_metrics:
        table.add_row([get_ticker_name(choice), choice_metrics['Trailing P/E Ratio'], choice_metrics['P/S Ratio'], choice_metrics['EV/EBITDA Ratio'], 
                    choice_metrics['Trailing PEG Ratio'], choice_metrics['Dividend Yield']])
    
    for ticker in tickers_list:
        peer_metrics = get_valuation_metrics(ticker)
        if peer_metrics:
            table.add_row([get_ticker_name(ticker), peer_metrics['Trailing P/E Ratio'], peer_metrics['P/S Ratio'], peer_metrics['EV/EBITDA Ratio'], 
                        peer_metrics['Trailing PEG Ratio'], peer_metrics['Dividend Yield']])
    
    return table

def get_profitability_metrics_table(choice, tickers_list):
    def get_profitability_metrics(ticker):
        try:
            ticker_info = yf.Ticker(ticker)
                      
            return {
                'Ticker': ticker,
                'ROE': "{:.2f}".format(ticker_info.info.get('returnOnEquity') * 100) if ticker_info.info.get('returnOnEquity') is not None else None,
                'ROA': "{:.2f}".format(ticker_info.info.get('returnOnAssets') * 100) if ticker_info.info.get('returnOnAssets') is not None else None,
                'Gross Margin': "{:.2f}".format(ticker_info.info.get('grossMargins') * 100) if ticker_info.info.get('grossMargins') is not None else None,
                'Operating Margin': "{:.2f}".format(ticker_info.info.get('operatingMargins') * 100)if ticker_info.info.get('operatingMargins') is not None else None,
                'Profit Margin': "{:.2f}".format(ticker_info.info.get('profitMargins') * 100) if ticker_info.info.get('profitMargins') is not None else None,
            }

        except Exception as e:
            print(f"Error occurred while fetching data for {ticker}: {e}")
            return None
    
    table = PrettyTable()
    table.field_names = ['Ticker', 'ROE Ratio', 'ROA Ratio', 'Gross Margin (%)', 'Operating Margin (%)', 'Profit Margin (%)']

    choice_metrics = get_profitability_metrics(choice)
    if choice_metrics:
        table.add_row([get_ticker_name(choice), choice_metrics['ROE'], choice_metrics['ROA'], 
                    choice_metrics['Gross Margin'], choice_metrics['Operating Margin'], 
                    choice_metrics['Profit Margin']])
    
    for ticker in tickers_list:
        peer_metrics = get_profitability_metrics(ticker)
        if peer_metrics:
            table.add_row([get_ticker_name(ticker), peer_metrics['ROE'], peer_metrics['ROA'], 
                        peer_metrics['Gross Margin'], peer_metrics['Operating Margin'], 
                        peer_metrics['Profit Margin']])
    
    return table

def get_leverage_metrics_table(choice, tickers_list):
    def get_leverage_metrics(ticker):
        try:
            ticker_info = yf.Ticker(ticker)
            balance_sheet = ticker_info.balance_sheet
            
            return {
                'Ticker': ticker,
                'Debt-to-Equity Ratio': "{:.2f}".format(ticker_info.info.get('debtToEquity')) if ticker_info.info.get('debtToEquity') is not None else None
            }
        except Exception as e:
            print(f"Error occurred while fetching data for {ticker}: {e}")
            return None
    
    table = PrettyTable()
    table.field_names = ['Ticker', 'Debt-to-Equity Ratio']
    
    choice_metrics = get_leverage_metrics(choice)
    if choice_metrics:
        table.add_row([get_ticker_name(choice), choice_metrics['Debt-to-Equity Ratio']])
    
    for ticker in tickers_list:
        peer_metrics = get_leverage_metrics(ticker)
        if peer_metrics:
            table.add_row([get_ticker_name(ticker), peer_metrics['Debt-to-Equity Ratio']])
    
    return table

def get_risk_metrics_table(choice, tickers_list):
    def get_risk_metrics(ticker):
        try:
            ticker_info = yf.Ticker(ticker)
            
            return {
                'Ticker': ticker,
                'Beta': "{:.2f}".format(ticker_info.info.get('beta')) if ticker_info.info.get('beta') is not None else None
            }
        except Exception as e:
            print(f"Error occurred while fetching data for {ticker}: {e}")
            return None

    table = PrettyTable()
    table.field_names = ['Ticker', 'Beta']
    
    choice_metrics = get_risk_metrics(choice)
    if choice_metrics:
        table.add_row([get_ticker_name(choice), choice_metrics['Beta']])
    
    for ticker in tickers_list:
        risk_metrics = get_risk_metrics(ticker)
        if risk_metrics:
            table.add_row([get_ticker_name(ticker), risk_metrics['Beta']])
    
    return table

def get_expected_annual_earnings_growth_table(choice, tickers_list):
    def get_expected_annual_earnings_growth(ticker):
        try:
            ticker_info = yf.Ticker(ticker)
            summary_info = ticker_info.info
        
            return {
                    'Ticker': ticker,
                    'Expected Annual Earnings Growth Rate': "{:.2f}".format(ticker_info.info.get('earningsGrowth')) if ticker_info.info.get('earningsGrowth') is not None else None,
                    'Expected Annual Revenue Growth Rate': "{:.2f}".format(ticker_info.info.get('revenueGrowth')) if ticker_info.info.get('revenueGrowth') is not None else None
            }
    
        except Exception as e:
            print(f"Error occurred while fetching data for {ticker}: {e}")
            return None
    
    table = PrettyTable()
    table.field_names = ['Ticker', 'Expected Annual Earnings Growth Rate (%)', 'Expected Annual Revenue Growth Rate (%)']
    
    choice_growth_rate = get_expected_annual_earnings_growth(choice)
    if choice_growth_rate:
        table.add_row([get_ticker_name(choice), choice_growth_rate['Expected Annual Earnings Growth Rate'], choice_growth_rate['Expected Annual Revenue Growth Rate']])

    for ticker in tickers_list:
        peer_growth_rate = get_expected_annual_earnings_growth(ticker)
        if peer_growth_rate:
            table.add_row([get_ticker_name(ticker), peer_growth_rate['Expected Annual Earnings Growth Rate'],  peer_growth_rate['Expected Annual Revenue Growth Rate']])
    
    return table

def get_stock_country_symbols(choice):
    stock = yf.Ticker(choice)
    max_attempts = 3
    attempt = 1

    while attempt <= max_attempts:
        try:
            df_list = pd.read_html('https://finance.yahoo.com/world-indices/')
            majorStockIdx = df_list[0]
            break  
        except Exception as e:
            print(f"Attempt {attempt}: An error occurred while fetching data: {e}")
            if attempt == max_attempts:
                print(f"Reached maximum number of attempts ({max_attempts}). Exiting.")
                majorStockIdx = None
                break  
            else:
                print("Retrying...")
                attempt += 1

    timezone = stock.info.get('timeZoneFullName')
    stock_country = timezone.split('/')[0].strip()

    symbols_by_country = {}

    for index, row in majorStockIdx.iterrows():
        symbol = row['Symbol']
        ticker = yf.Ticker(symbol)
        
        timezone = ticker.info.get('timeZoneFullName')
        country = timezone.split('/')[0].strip()
        
        if country == "Australia" or country == "Pacific":
            country = "Australia/Pacific"

        symbols_by_country.setdefault(country, []).append(symbol)

    stock_country_symbols = symbols_by_country.get(stock_country, [])
    
    return stock_country_symbols

def get_index_info(stock_country_symbols):
    table = PrettyTable()
    table.field_names = ["Index", "Name", "Currency", "Exchange", "52 Week Range"]
    
    for symbol in stock_country_symbols:
        ticker = yf.Ticker(symbol)

        index = symbol
        name = get_ticker_name(symbol)
        currency = ticker.info.get('currency', 'None')
        exchange = ticker.info.get('exchange', 'None')
        week_range = str(format_large_number(ticker.info.get('fiftyTwoWeekLow', 0))) + " - " + str(format_large_number(ticker.info.get('fiftyTwoWeekHigh', 0)))

        table.add_row([index, name, currency, exchange, week_range])

    return table

def calculate_performance_metrics(stock_ticker):
    start_date = (datetime.datetime.now() - datetime.timedelta(days=365)).strftime('%Y-%m-%d')
    end_date = datetime.datetime.now().strftime('%Y-%m-%d')
    
    stock_data = yf.download(stock_ticker, start=start_date, end=end_date)

    if stock_data.empty:
        print(f"No data available for {stock_ticker} within the specified time frame.")
        return None
    
    stock_returns = stock_data['Adj Close'].pct_change().dropna()
    total_return = (stock_data['Adj Close'].iloc[-1] / stock_data['Adj Close'].iloc[0]) - 1
    annualized_return = ((1 + total_return) ** (1 / len(stock_returns))) - 1
    volatility = stock_returns.std()
    max_drawdown = (stock_data['Adj Close'].min() / stock_data['Adj Close'].max()) - 1
    
    return {
        'Total Return': "{:.2f}".format(total_return),
        'Annualized Return': "{:.2f}".format(annualized_return),
        'Volatility': "{:.2f}".format(volatility),
        'Maximum Drawdown': "{:.2f}".format(max_drawdown)
    }

def calculate_market_metrics(stock_ticker, index_ticker):
    start_date = (datetime.datetime.now() - datetime.timedelta(days=365)).strftime('%Y-%m-%d')
    end_date = datetime.datetime.now().strftime('%Y-%m-%d')
    
    stock_data = yf.download(stock_ticker, start=start_date, end=end_date)
    index_data = yf.download(index_ticker, start=start_date, end=end_date)
    
    if stock_data.empty or index_data.empty:
        print("No data available for the specified tickers within the specified time frame.")
        return None
    
    index_market_cap = index_data['Adj Close'].iloc[-1] * index_data['Volume'].iloc[-1]

    stock_returns = stock_data['Adj Close'].pct_change().dropna()
    index_returns = index_data['Adj Close'].pct_change().dropna()
    beta = stock_returns.cov(index_returns) / index_returns.var()
    
    stock_returns, index_returns = stock_returns.align(index_returns, join="inner")
    correlation = stock_returns.corr(index_returns)

    index_avg_volume = index_data['Volume'].mean()
    
    return {
        'Market Capitalization': format_large_number("{:.2f}".format(index_market_cap)),
        'Beta': "{:.2f}".format(beta),
        'Correlation': "{:.2f}".format(correlation),
        'Liquidity (Avg. Daily Volume)': format_large_number("{:.2f}".format(index_avg_volume))
    }

def calculate_sma(stock_ticker, window):
    start_date = (datetime.datetime.now() - datetime.timedelta(days=365 * 2)).strftime('%Y-%m-%d')
    end_date = datetime.datetime.now().strftime('%Y-%m-%d')

    stock_data = yf.download(stock_ticker, start=start_date, end=end_date)
    
    if stock_data.empty:
        print(f"No price data available for {stock_ticker} within the specified time frame.")
        return None
    
    sma = stock_data['Adj Close'].rolling(window=window).mean()
    return sma

def get_fifty_day_average(stock_ticker):
    stock_info = yf.Ticker(stock_ticker)
    fifty_day_average = stock_info.info.get('fiftyDayAverage', None)
    return format_large_number("{:.2f}".format(fifty_day_average)) if fifty_day_average is not None else 'N/A'

def get_two_hundred_day_average(stock_ticker):
    stock_info = yf.Ticker(stock_ticker)
    two_hundred_day_average = stock_info.info.get('twoHundredDayAverage', None)
    return format_large_number("{:.2f}".format(two_hundred_day_average)) if two_hundred_day_average is not None else 'N/A'

def index(request):
    if request.method == 'POST':
        form = StockForm(request.POST)
            
        if form.is_valid():
            symbol = form.cleaned_data['symbol']

            stock_data = yf.Ticker(symbol)
            stock_info = stock_data.info

            name = stock_info.get('longName', 'N/A')
            sector = stock_info.get('sector', 'N/A')
       
            # Save form data in the session
            request.session['symbol'] = symbol
            request.session['name'] = name

            Stock.objects.filter(symbol=symbol).update_or_create(
                                                                    symbol=symbol, defaults =  {
                                                                                                'name': name, 
                                                                                                'sector': sector                                    
                                                                                                }
                                                                )
            return redirect('profile')
        else:
            print("Form errors:")
    else:
        form = StockForm()
        context = {'form': form} 

    return render(request, 'stocks/index.html', context)

def profile(request):
    symbol = request.session.get('symbol', 'N/A')
    stock = yf.Ticker(symbol)

    company_info = {
        'name': stock.info.get('longName'),
        'address': ", ".join(filter(None, [stock.info.get('address1'), stock.info.get('city'), stock.info.get('state'), stock.info.get('zip'), stock.info.get('country')])),
        'phone': stock.info.get('phone'),
        'website': stock.info.get('website'),
        'exchange': stock.info.get('exchange'),
        'currency': stock.info.get('currency'),
        'employees': stock.info.get('fullTimeEmployees'),
        'timezone': stock.info.get('timeZoneFullName'),
        'industry': stock.info.get('industry'),
        'sector': stock.info.get('sector'),
        'description': stock.info.get('longBusinessSummary')
    }

    company_officers = stock.info.get('companyOfficers', [])
    officers_table = PrettyTable(['Name', 'Title'])
    for officer in company_officers:
        officers_table.add_row([officer.get('name'), officer.get('title')])
    
    return render(request, 'stocks/profile.html', {'company_info': company_info, 
                                                   'officers_table': officers_table})

def charts(request):
    symbol = request.session.get('symbol', 'N/A')
    stock = yf.Ticker(symbol)

    plot1_html = generate_stock_price_chart(symbol, '1d', '1 day')   

    top_5_peers = get_top_n_peers(symbol, 5)
    stock_tickers_1 = [symbol] + top_5_peers
    plot2_html = plot_stock_peer_price_change_percentage(stock_tickers_1, '1d', '1 day')

    stock_country_symbols = get_stock_country_symbols(symbol)
    top_5_indices = get_top_n_indices(stock_country_symbols, 5, 'max')
    stock_tickers_2 = [symbol] + top_5_indices
    plot3_html = plot_stock_index_price_change_percentage(stock_tickers_2, '1d', '1 day')

    window_50 = 50
    window_200 = 200
    data = fetch_data(symbol, ALPHA_VANTAGE_API_KEY, 'daily')

    try:
        sma_50 = calculate_custom_sma(data, window_50, 'Time Series (Daily)')
        sma_200 = calculate_custom_sma(data, window_200, 'Time Series (Daily)')

        if sma_50 is not None and sma_200 is not None:
            plot4_html = plot_prices_with_sma(symbol, data, sma_50, sma_200, 'Daily', 'Time Series (Daily)')
        else:
            print("Error: Unable to calculate one or both SMAs.")
            plot4_html = None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        plot4_html = None

    plot5_html = plot_bollinger_bands(symbol, ALPHA_VANTAGE_API_KEY, 'daily', 'Daily')

    rsi_df = get_rsi_data(symbol, POLYGON_API_KEY, 'day')
    plot6_html = plot_rsi(symbol, rsi_df, 'Daily')

    if request.method == 'GET':
        chart_type = request.GET.get('chart_type') 

        selected_time_period_1 = request.GET.get('selected_time_period_1', '1d')
        selected_slider_value_1 = request.GET.get('selected_slider_value_1', '1 day') 

        selected_time_period_2 = request.GET.get('selected_time_period_2', '1d')
        selected_slider_value_2 = request.GET.get('selected_slider_value_2', '1 day') 

        selected_time_period_3 = request.GET.get('selected_time_period_3', '1d')  
        selected_slider_value_3 = request.GET.get('selected_slider_value_3', '1 day') 

        selected_time_period_4 = request.GET.get('selected_time_period_4', 'daily')  
        selected_slider_value_4 = request.GET.get('selected_slider_value_4', 'Daily') 

        selected_time_period_5 = request.GET.get('selected_time_period_5', 'daily')  
        selected_slider_value_5 = request.GET.get('selected_slider_value_5', 'Daily') 

        selected_time_period_6 = request.GET.get('selected_time_period_6', 'day')  
        selected_slider_value_6 = request.GET.get('selected_slider_value_6', 'Daily') 
        
        return render(request, 'stocks/charts.html',    {'plot1_html': plot1_html,
                                                        'plot2_html': plot2_html,
                                                        'plot3_html': plot3_html,
                                                        'plot4_html': plot4_html,
                                                        'plot5_html': plot5_html,
                                                        'plot6_html': plot6_html})
    elif request.method == 'POST':
        chart_type = request.POST.get('chart_type') 
        
        if chart_type == 'stock_price':
            selected_time_period_1 = request.POST.get('selected_time_period_1', '1d')  
            selected_slider_value_1 = request.POST.get('selected_slider_value_1', '1 day') 

            plot1_html = generate_stock_price_chart(symbol, selected_time_period_1, selected_slider_value_1)

        elif chart_type == 'peer_price_percentage_change':
            selected_time_period_2 = request.POST.get('selected_time_period_2', '1d')  
            selected_slider_value_2 = request.POST.get('selected_slider_value_2', '1 day') 

            plot2_html = plot_stock_peer_price_change_percentage(stock_tickers_1, selected_time_period_2, selected_slider_value_2)

        elif chart_type == 'index_price_percentage_change':
            selected_time_period_3 = request.POST.get('selected_time_period_3', '1d')  
            selected_slider_value_3 = request.POST.get('selected_slider_value_3', '1 day') 

            plot3_html = plot_stock_index_price_change_percentage(stock_tickers_2, selected_time_period_3, selected_slider_value_3)

        elif chart_type == 'simple_moving_averages_chart':
            selected_time_period_4 = request.POST.get('selected_time_period_4', 'daily')  
            selected_slider_value_4 = request.POST.get('selected_slider_value_4', 'Daily') 

            if selected_time_period_4 == 'daily':
                time_series_key = 'Time Series (Daily)'
            elif selected_time_period_4 == 'weekly' or selected_time_period_4 == 'monthly':
                time_series_key = f'{selected_slider_value_4} Time Series'

            data = fetch_data(symbol, ALPHA_VANTAGE_API_KEY, selected_time_period_4)
            sma_50 = calculate_custom_sma(data, window_50, time_series_key)
            sma_200 = calculate_custom_sma(data, window_200, time_series_key)

            plot4_html = plot_prices_with_sma(symbol, data, sma_50, sma_200, selected_slider_value_4, time_series_key)

        elif chart_type == 'bollinger_bands_chart':
            selected_time_period_5 = request.POST.get('selected_time_period_5', 'daily')  
            selected_slider_value_5 = request.POST.get('selected_slider_value_5', 'Daily') 

            plot5_html = plot_bollinger_bands(symbol, ALPHA_VANTAGE_API_KEY, selected_time_period_5, selected_slider_value_5)

        elif chart_type == 'relative_strength_index_chart':
            selected_time_period_6 = request.POST.get('selected_time_period_6', 'day')  
            selected_slider_value_6 = request.POST.get('selected_slider_value_6', 'Daily') 

            rsi_df = get_rsi_data(symbol, POLYGON_API_KEY, selected_time_period_6)
            plot6_html = plot_rsi(symbol, rsi_df, selected_slider_value_6)

        return render(request, 'stocks/charts.html',    {'plot1_html': plot1_html,
                                                        'plot2_html': plot2_html,
                                                        'plot3_html': plot3_html,
                                                        'plot4_html': plot4_html,
                                                        'plot5_html': plot5_html,
                                                        'plot6_html': plot6_html})
           
    return render(request, 'stocks/charts.html', {})

def news(request):
    symbol = request.session.get('symbol', 'N/A')

    news_data = []
    news_items = fetch_news(symbol, POLYGON_API_KEY)

    if news_items and 'results' in news_items:
        for item in news_items['results']:
            if 'article_url' in item:
                article_url = item['article_url']
                article_text = read_article(article_url)
                if article_text:
                    preprocessed_text = preprocess_text(article_text)
                    sentiment_score, sentiment = get_sentiment(preprocessed_text)
                    news_data.append({'Article_URL': article_url, 'Article_Title': item['title'], 'Sentiment_Score': sentiment_score, 'Sentiment': sentiment})
                else:
                    print("Failed to read article from URL:", article_url)
    else:
        print("Failed to fetch news items or results not found.")

    news_df = pd.DataFrame(news_data)
    n_samples = min(5, len(news_df))
    news_df = news_df.sample(n=n_samples, replace=False, random_state=None)
    news_df.reset_index(drop=True, inplace=True)

    return render(request, 'stocks/news.html', {'news_df': news_df})

def insights(request):
    symbol = request.session.get('symbol', 'N/A')
    stock = yf.Ticker(symbol)

    basic_information = {
        'market_cap': format_large_number(stock.info.get('marketCap')),
        'previous_close': stock.info.get('previousClose'),
        'current_price': stock.info.get('currentPrice'),
        'fifty_two_week_range': f"{stock.info.get('fiftyTwoWeekLow')}-{stock.info.get('fiftyTwoWeekHigh')}",
        'average_volume': format_large_number(stock.info.get('averageVolume')),
    }

    dividends_and_splits = {
        'dividend_rate': stock.info.get('dividendRate'),
        'payout_ratio': stock.info.get('payoutRatio'),
        'dividend_yield': stock.info.get('dividendYield'),
        'last_dividend_value': stock.info.get('lastDividendValue'),
        'ex_dividend_date': datetime.datetime.fromtimestamp(stock.info.get('exDividendDate')).strftime('%d/%m/%Y') if stock.info.get('exDividendDate') else None,
        'last_split_factor': stock.info.get('lastSplitFactor'),
        'last_split_date': datetime.datetime.fromtimestamp(stock.info.get('lastSplitDate')).strftime('%d/%m/%Y')  if stock.info.get('lastSplitDate') else None,
        'last_dividend_date': datetime.datetime.fromtimestamp(stock.info.get('lastDividendDate')).strftime('%d/%m/%Y')  if stock.info.get('lastDividendDate') else None,
        'five_year_average_dividend_yield': stock.info.get('fiveYearAvgDividendYield'),
    }

    earnings_and_revenue = {
        'trailing_eps': stock.info.get('trailingEps'),
        'forward_eps': stock.info.get('forwardEps'),
        'total_revenue': format_large_number(stock.info.get('totalRevenue')),
        'earnings_growth': stock.info.get('earningsGrowth'),
        'earnings_quarterly_growth': stock.info.get('earningsQuarterlyGrowth'),
        'revenue_growth': stock.info.get('revenueGrowth'),
    }

    ratios_and_margins = {
        'trailing_pe_ratio': stock.info.get('trailingPE'),
        'forward_pe_ratio': stock.info.get('forwardPE'),
        'profit_margins': stock.info.get('profitMargins'),
        'return_on_equity': stock.info.get('returnOnEquity'),
        'return_on_assets': stock.info.get('returnOnAssets'),
        'debt_to_equity_ratio': stock.info.get('debtToEquity'),
        'gross_margins': stock.info.get('grossMargins'),
        'ebitda_margins': stock.info.get('ebitdaMargins'),
        'operating_margins': stock.info.get('operatingMargins'),
        'trailing_peg_ratio': stock.info.get('trailingPegRatio'),
        'peg_ratio': stock.info.get('pegRatio'),
    }

    cash_flow_and_enterprise_value = {
        'total_cash': format_large_number(stock.info.get('totalCash')),
        'free_cash_flow': format_large_number(stock.info.get('freeCashflow')),
        'ebitda': format_large_number(stock.info.get('ebitda')),
        'operating_cash_flow': format_large_number(stock.info.get('operatingCashflow')),
        'enterprise_value': format_large_number(stock.info.get('enterpriseValue')),
        'enterprise_to_revenue_ratio': stock.info.get('enterpriseToRevenue'),
        'enterprise_to_ebitda_ratio': stock.info.get('enterpriseToEbitda'),
    }

    other_metrics = {
        'book_value': stock.info.get('bookValue'),
        'price_to_book_ratio': stock.info.get('priceToBook'),
        'revenue_per_share': stock.info.get('revenuePerShare'),
        'total_cash_per_share': stock.info.get('totalCashPerShare'),
        'average_daily_volume_10_day': format_large_number(stock.info.get('averageDailyVolume10Day')),
        'regular_market_volume': format_large_number(stock.info.get('regularMarketVolume')),
        'regular_market_day_low': stock.info.get('regularMarketDayLow'),
        'regular_market_day_high': stock.info.get('regularMarketDayHigh'),
        'regular_market_open': stock.info.get('regularMarketOpen'),
        'regular_market_previous_close': stock.info.get('regularMarketPreviousClose'),
    }

    return render(request, 'stocks/insights.html', {'basic_information': basic_information,
                                                    'dividends_and_splits': dividends_and_splits,
                                                    'earnings_and_revenue': earnings_and_revenue,
                                                    'ratios_and_margins': ratios_and_margins,
                                                    'cash_flow_and_enterprise_value': cash_flow_and_enterprise_value,
                                                    'other_metrics': other_metrics})

def analysts(request):
    symbol = request.session.get('symbol', 'N/A')
    stock = yf.Ticker(symbol)

    analyst_recommendations = {
        "number_of_analyst_opinions": stock.info.get('numberOfAnalystOpinions'),
        "recommendation_mean": stock.info.get('recommendationMean'),
        "recommendation_key": stock.info.get('recommendationKey'),
        "target_mean_price": stock.info.get('targetMeanPrice'),
        "target_high_price": stock.info.get('targetHighPrice'),
        "target_low_price": stock.info.get('targetLowPrice'),
        "target_median_price": stock.info.get('targetMedianPrice')
    }

    recommendations = stock.recommendations
    current_date = datetime.datetime.today()
    current_month = current_date.strftime('%B')

    months = ['December', 'November', 'October', 'September', 'August', 'July', 'June', 'May', 'April', 'March', 'February', 'January']

    start_index = months.index(current_month)

    rotated_months = deque(months)
    rotated_months.rotate(-start_index)

    custom_labels = list(rotated_months)

    trace_strong_buy = go.Bar(x=custom_labels, y=recommendations['strongBuy'], name='Strong Buy')
    trace_buy = go.Bar(x=custom_labels, y=recommendations['buy'], name='Buy')
    trace_hold = go.Bar(x=custom_labels, y=recommendations['hold'], name='Hold')
    trace_sell = go.Bar(x=custom_labels, y=recommendations['sell'], name='Sell')
    trace_strong_sell = go.Bar(x=custom_labels, y=recommendations['strongSell'], name='Strong Sell')

    layout = go.Layout  (title='Analyst Recommendations Chart',
                        xaxis=dict(title='Month'),
                        yaxis=dict(title='Number of Recommendations'),
                        height=400,
                        width=800)

    fig = go.Figure(data=[trace_strong_buy, trace_buy, trace_hold, trace_sell, trace_strong_sell], layout=layout)
    plot1_html = pio.to_html(fig, full_html=False)

    dates = list(stock.calendar.keys())
    earnings = list(stock.calendar.values())

    earnings_table = PrettyTable()
    earnings_table.field_names = ["Date", "Earnings"]

    earning_dates = []
    for date, earning in zip(dates, earnings):
        if isinstance(earning, list):
            for element in earning:
                if isinstance(element, datetime.date):
                    earning_dates.append(element.strftime("%d/%m/%Y"))
                else:
                    earning_dates.append(str(element)) 
            earnings_table.add_row([date, ", ".join(earning_dates)])
        elif isinstance(earning, datetime.date):
            earnings_table.add_row([date, earning.strftime("%d/%m/%Y")])
        elif isinstance(earning, int):
            earnings_table.add_row([date, format_large_number(earning)])
        else:
            earnings_table.add_row([date, str(earning)])

        balance_sheet = stock.balance_sheet
        date_strings = list(balance_sheet.keys())
        financials_components = {extract_date(date): {} for date in date_strings}
        components = ['Total Liabilities Net Minority Interest', 'Total Assets']

        for date in date_strings:
            for component in components:
                try:
                    financials_components[extract_date(date)][component] = balance_sheet[pd.Timestamp(date)][component]
                except KeyError:
                    financials_components[extract_date(date)][component] = None

        dates = [extract_date(date) for date in date_strings]
        years = [date.strftime('%Y') for date in date_strings]
        liabilities_data = [financials_components[date]['Total Liabilities Net Minority Interest'] for date in dates]
        assets_data = [financials_components[date]['Total Assets'] for date in dates]

        trace1 = go.Bar(
            x=years,
            y=liabilities_data,
            name='Total Liabilities'
        )

        trace2 = go.Bar(
            x=years,
            y=assets_data,
            name='Total Assets'
        )

        data = [trace1, trace2]

        layout = go.Layout(
            barmode='group',
            title='Total Liabilities and Total Assets Over Time (Annual)',
            xaxis=dict(title='Year'),
            yaxis=dict(title='Amount'),
            height=400,
            width=800
        )

        fig = go.Figure(data=data, layout=layout)
        plot2_html = pio.to_html(fig, full_html=False)

        income_statement = stock.income_stmt
        date_strings = list(income_statement.keys())
        financials_components = {extract_date(date): {} for date in date_strings}
        components = ['Net Income', 'Total Revenue']

        for date in date_strings:
            for component in components:
                try:
                    financials_components[extract_date(date)][component] = income_statement[pd.Timestamp(date)][component]
                except KeyError:
                    financials_components[extract_date(date)][component] = None
        
        dates = [extract_date(date) for date in date_strings]
        years = [date.strftime('%Y') for date in date_strings]
        net_income_data = [financials_components[date]['Net Income'] for date in dates]
        revenue_data = [financials_components[date]['Total Revenue'] for date in dates]

        trace1 = go.Bar(
            x=years,
            y=net_income_data,
            name='Net Income'
        )

        trace2 = go.Bar(
            x=years,
            y=revenue_data,
            name='Total Revenue'
        )

        data = [trace1, trace2]

        layout = go.Layout(
            barmode='group',
            title='Net Income and Total Revenue Over Time (Annual)',
            xaxis=dict(title='Year'),
            yaxis=dict(title='Amount'),
            height=400,
            width=800
        )

        fig = go.Figure(data=data, layout=layout)
        plot3_html = pio.to_html(fig, full_html=False)

        cashflow = stock.cashflow
        date_strings = list(cashflow.keys())
        financials_components = {extract_date(date): {} for date in date_strings}
        components = ['Changes In Cash']

        for date in date_strings:
            for component in components:
                try:
                    financials_components[extract_date(date)][component] = cashflow[pd.Timestamp(date)][component]
                except KeyError:
                    financials_components[extract_date(date)][component] = None

        dates = [extract_date(date) for date in date_strings]
        years = [date.strftime('%Y') for date in date_strings]
        net_changes_in_cash_data = [financials_components[date]['Changes In Cash'] for date in dates]

        trace = go.Bar(
            x=years,
            y=net_changes_in_cash_data,
            name='Net Changes In Cash'
        )

        layout = go.Layout(
            title='Net Changes In Cash Over Time (Annual)',
            xaxis=dict(title='Year'),
            yaxis=dict(title='Amount'),
            height=400,
            width=800
        )

        fig = go.Figure(data=[trace], layout=layout)
        plot4_html = pio.to_html(fig, full_html=False)
    
    if request.method == 'GET':
        chart_type = request.GET.get('chart_type') 
        period = 'Annual'

        return render(request, 'stocks/analysts.html',  {'period': period,
                                                        'analyst_recommendations': analyst_recommendations,
                                                        'plot1_html': plot1_html,
                                                        'earnings_table': earnings_table,
                                                        'plot2_html': plot2_html,
                                                        'plot3_html': plot3_html,
                                                        'plot4_html': plot4_html})
    elif request.method == 'POST':
        chart_type = request.POST.get('chart_type') 

        if chart_type == 'financials-balance-sheet':
            action = request.POST.get('action', 'annual')
         
            if action == 'annual':
                balance_sheet = stock.balance_sheet
                period = 'Annual'

                date_strings = list(balance_sheet.keys())
                financials_components = {extract_date(date): {} for date in date_strings}
                components = ['Total Liabilities Net Minority Interest', 'Total Assets']

                for date in date_strings:
                    for component in components:
                        try:
                            financials_components[extract_date(date)][component] = balance_sheet[pd.Timestamp(date)][component]
                        except KeyError:
                            financials_components[extract_date(date)][component] = None

                dates = [extract_date(date) for date in date_strings]
                years = [date.strftime('%Y') for date in date_strings]
                liabilities_data = [financials_components[date]['Total Liabilities Net Minority Interest'] for date in dates]
                assets_data = [financials_components[date]['Total Assets'] for date in dates]

                trace1 = go.Bar(
                    x=years,
                    y=liabilities_data,
                    name='Total Liabilities'
                )

                trace2 = go.Bar(
                    x=years,
                    y=assets_data,
                    name='Total Assets'
                )

                data = [trace1, trace2]

                layout = go.Layout(
                    barmode='group',
                    title='Total Liabilities and Total Assets Over Time (Annual)',
                    xaxis=dict(title='Year'),
                    yaxis=dict(title='Amount'),
                    height=400,
                    width=800
                )

                fig = go.Figure(data=data, layout=layout)
                plot2_html = pio.to_html(fig, full_html=False)
            elif action == 'quarterly':
                balance_sheet = stock.quarterly_balance_sheet
                period = 'Quarterly'

                date_strings = list(balance_sheet.keys())
                financials_components = {extract_date(date): {} for date in date_strings}
                components = ['Total Liabilities Net Minority Interest', 'Total Assets']

                for date in date_strings:
                    for component in components:
                        try:
                            financials_components[extract_date(date)][component] = balance_sheet[pd.Timestamp(date)][component]
                        except KeyError:
                            financials_components[extract_date(date)][component] = None

                dates = [extract_date(date) for date in date_strings]
                quarters = [f"{(date.month - 1) // 3 + 1}Q{date.strftime('%Y')}" for date in date_strings]
                liabilities_data = [financials_components[date]['Total Liabilities Net Minority Interest'] for date in dates]
                assets_data = [financials_components[date]['Total Assets'] for date in dates]

                trace1 = go.Bar(
                    x=quarters,
                    y=liabilities_data,
                    name='Total Liabilities'
                )

                trace2 = go.Bar(
                    x=quarters,
                    y=assets_data,
                    name='Total Assets'
                )

                data = [trace1, trace2]

                layout = go.Layout(
                    barmode='group',
                    title='Total Liabilities and Total Assets Over Time (Quarterly)',
                    xaxis=dict(title='Quarter'),
                    yaxis=dict(title='Amount'),
                    height=400,
                    width=800
                )

                fig = go.Figure(data=data, layout=layout)
                plot2_html = pio.to_html(fig, full_html=False)
        elif chart_type == 'financials-income-statement':
            action = request.POST.get('action', 'annual')
          
            if action == 'annual':
                income_statement = stock.income_stmt
                period = 'Annual'

                date_strings = list(income_statement.keys())
                financials_components = {extract_date(date): {} for date in date_strings}
                components = ['Net Income', 'Total Revenue']

                for date in date_strings:
                    for component in components:
                        try:
                            financials_components[extract_date(date)][component] = income_statement[pd.Timestamp(date)][component]
                        except KeyError:
                            financials_components[extract_date(date)][component] = None
                
                dates = [extract_date(date) for date in date_strings]
                years = [date.strftime('%Y') for date in date_strings]
                net_income_data = [financials_components[date]['Net Income'] for date in dates]
                revenue_data = [financials_components[date]['Total Revenue'] for date in dates]

                trace1 = go.Bar(
                    x=years,
                    y=net_income_data,
                    name='Net Income'
                )

                trace2 = go.Bar(
                    x=years,
                    y=revenue_data,
                    name='Total Revenue'
                )

                data = [trace1, trace2]

                layout = go.Layout(
                    barmode='group',
                    title='Net Income and Total Revenue Over Time (Annual)',
                    xaxis=dict(title='Year'),
                    yaxis=dict(title='Amount'),
                    height=400,
                    width=800
                )

                fig = go.Figure(data=data, layout=layout)
                plot3_html = pio.to_html(fig, full_html=False)
            elif action == 'quarterly':
                income_statement = stock.quarterly_income_stmt
                period = 'Quarterly'

                date_strings = list(income_statement.keys())
                financials_components = {extract_date(date): {} for date in date_strings}
                components = ['Net Income', 'Total Revenue']

                for date in date_strings:
                    for component in components:
                        try:
                            financials_components[extract_date(date)][component] = income_statement[pd.Timestamp(date)][component]
                        except KeyError:
                            financials_components[extract_date(date)][component] = None

                dates = [extract_date(date) for date in date_strings]
                quarters = [f"{(date.month - 1) // 3 + 1}Q{date.strftime('%Y')}" for date in date_strings]
                net_income_data = [financials_components[date]['Net Income'] for date in dates]
                revenue_data = [financials_components[date]['Total Revenue'] for date in dates]

                trace1 = go.Bar(
                    x=quarters,
                    y=net_income_data,
                    name='Net Income'
                )

                trace2 = go.Bar(
                    x=quarters,
                    y=revenue_data,
                    name='Total Revenue'
                )

                data = [trace1, trace2]

                layout = go.Layout(
                    barmode='group',
                    title='Net Income and Total Revenue Over Time (Quarterly)',
                    xaxis=dict(title='Quarter'),
                    yaxis=dict(title='Amount'),
                    height=400,
                    width=800
                )

                fig = go.Figure(data=data, layout=layout)
                plot3_html = pio.to_html(fig, full_html=False)
        elif chart_type == 'financials-cashflow':
            action = request.POST.get('action', 'annual')
           
            if action == 'annual':
                cashflow = stock.cashflow
                period = 'Annual'

                date_strings = list(cashflow.keys())
                financials_components = {extract_date(date): {} for date in date_strings}
                components = ['Changes In Cash']

                for date in date_strings:
                    for component in components:
                        try:
                            financials_components[extract_date(date)][component] = cashflow[pd.Timestamp(date)][component]
                        except KeyError:
                            financials_components[extract_date(date)][component] = None

                dates = [extract_date(date) for date in date_strings]
                years = [date.strftime('%Y') for date in date_strings]
                net_changes_in_cash_data = [financials_components[date]['Changes In Cash'] for date in dates]

                trace = go.Bar(
                    x=years,
                    y=net_changes_in_cash_data,
                    name='Net Changes In Cash'
                )

                layout = go.Layout(
                    title='Net Changes In Cash Over Time (Annual)',
                    xaxis=dict(title='Year'),
                    yaxis=dict(title='Amount'),
                    height=400,
                    width=800
                )

                fig = go.Figure(data=[trace], layout=layout)
                plot4_html = pio.to_html(fig, full_html=False)
            elif action == 'quarterly':
                cashflow = stock.quarterly_cashflow
                period = 'Quarterly'

                date_strings = list(cashflow.keys())
                financials_components = {extract_date(date): {} for date in date_strings}
                components = ['Changes In Cash']

                for date in date_strings:
                    for component in components:
                        try:
                            financials_components[extract_date(date)][component] = cashflow[pd.Timestamp(date)][component]
                        except KeyError:
                            financials_components[extract_date(date)][component] = None

                dates = [extract_date(date) for date in date_strings]
                quarters = [f"{(date.month - 1) // 3 + 1}Q{date.strftime('%Y')}" for date in date_strings]
                net_changes_in_cash_data = [financials_components[date]['Changes In Cash'] for date in dates]

                trace = go.Bar(
                    x=quarters,
                    y=net_changes_in_cash_data,
                    name='Net Changes In Cash'
                )

                layout = go.Layout(
                    title='Net Changes In Cash Over Time (Quarterly)',
                    xaxis=dict(title='Quarter'),
                    yaxis=dict(title='Amount'),
                    height=400,
                    width=800
                )

                fig = go.Figure(data=[trace], layout=layout)
                plot4_html = pio.to_html(fig, full_html=False)
                
        return render(request, 'stocks/analysts.html',  {'period': period,
                                                        'analyst_recommendations': analyst_recommendations,
                                                        'plot1_html': plot1_html,
                                                        'earnings_table': earnings_table,
                                                        'plot2_html': plot2_html,
                                                        'plot3_html': plot3_html,
                                                        'plot4_html': plot4_html})
    return render(request, 'stocks/analysts.html', {})

def risks(request):
    symbol = request.session.get('symbol', 'N/A')
    stock = yf.Ticker(symbol)

    market_volatility_and_performance = {
        'beta': stock.info.get('beta'),
        '52_week_change': stock.info.get('52WeekChange'),
        'sandp_52_week_change': stock.info.get('SandP52WeekChange'),
    }

    short_interest_and_ownership = {
        'short_percent_of_float': stock.info.get('shortPercentOfFloat'),
        'shares_short': stock.info.get('sharesShort'),
        'shares_percent_shares_out': stock.info.get('sharesPercentSharesOut'),
        'shares_short_prior_month': format_large_number(stock.info.get('sharesShortPriorMonth')),
        'shares_short_prior_month_date': datetime.datetime.fromtimestamp(stock.info.get('sharesShortPriorMonthDate')).strftime('%d/%m/%Y') if stock.info.get('sharesShortPriorMonthDate') else None,
        'shares_short_previous_month_date': datetime.datetime.fromtimestamp(stock.info.get('sharesShortPreviousMonthDate')).strftime('%d/%m/%Y') if stock.info.get('sharesShortPreviousMonthDate') else None,
        'held_percent_insiders': stock.info.get('heldPercentInsiders'),
        'held_percent_institutions': stock.info.get('heldPercentInstitutions'),
        'institutional_ownership': stock.info.get('institutionalOwnership'),
    }

    governance_and_audit_risk = {
        'governance_epoch_date': datetime.datetime.fromtimestamp(stock.info.get('governanceEpochDate')).strftime('%d/%m/%Y') if stock.info.get('governanceEpochDate') else None,
        'compensation_as_of_epoch_date': datetime.datetime.fromtimestamp(stock.info.get('compensationAsOfEpochDate')).strftime('%d/%m/%Y') if stock.info.get('compensationAsOfEpochDate') else None,
        'audit_risk': stock.info.get('auditRisk'),
        'board_risk': stock.info.get('boardRisk'),
        'compensation_risk': stock.info.get('compensationRisk'),
        'shareholder_rights_risk': stock.info.get('shareHolderRightsRisk'),
        'overall_risk': stock.info.get('overallRisk'),
    }

    return render(request, 'stocks/risks.html', {'market_volatility_and_performance': market_volatility_and_performance,
                                                'short_interest_and_ownership': short_interest_and_ownership,
                                                'governance_and_audit_risk': governance_and_audit_risk})

def financials_balance_sheet(request):
    symbol = request.session.get('symbol', 'N/A')
    stock = yf.Ticker(symbol)

    if request.method == 'GET':
        action = request.GET.get('action', 'annual')

        balance_sheet = stock.balance_sheet
        period = 'Annual'

        date_string = list(balance_sheet.keys())

        field_names = ['Component'] + [extract_date(date) for date in date_string]
        main_components = {extract_date(date): {} for date in date_string}
        components = ['Total Capitalization', 'Common Stock Equity', 'Capital Lease Obligations', 'Net Tangible Assets', 'Working Capital',
                        'Invested Capital', 'Tangible Book Value', 'Total Debt', 'Net Debt', 'Share Issued', 'Ordinary Shares Number', 
                        'Treasury Shares Number']
        
        for date in date_string:
            for component in components:
                try:
                    main_components[extract_date(date)][component] = format_large_number(balance_sheet[pd.Timestamp(date)][component])
                except KeyError:
                    main_components[extract_date(date)][component] = None

        main_components_table = PrettyTable()
        main_components_table.field_names = field_names

        for component in components:
            data_row = [component] + [main_components[date][component] for date in main_components]
            main_components_table.add_row(data_row)

        field_names = ['Component'] + [extract_date(date) for date in date_string]
        current_assets = {extract_date(date): {} for date in date_string}
        long_term_assets = {extract_date(date): {} for date in date_string}

        current_assets_components = ['Cash And Cash Equivalents', 'Other Short Term Investments', 'Accounts Receivable', 
                                    'Inventory', 'Hedging Assets Current', 'Other Current Assets']
        long_term_assets_components = ['Net PPE', 'Goodwill', 'Other Intangible Assets', 'Investments And Advances', 'Other Non Current Assets']

        for date in date_string:
            for component in current_assets_components:
                try:
                    current_assets[extract_date(date)][component] = format_large_number(balance_sheet[pd.Timestamp(date)][component])
                except KeyError:
                    current_assets[extract_date(date)][component] = None
            
            for component in long_term_assets_components:
                try:
                    long_term_assets[extract_date(date)][component] = format_large_number(balance_sheet[pd.Timestamp(date)][component])
                except KeyError:
                    long_term_assets[extract_date(date)][component] = None

        current_assets_table = PrettyTable()
        current_assets_table.field_names = field_names

        for component in current_assets_components:
            data_row = [component] + [current_assets[date][component] for date in current_assets]
            current_assets_table.add_row(data_row)

        long_term_assets_table = PrettyTable()
        long_term_assets_table.field_names = field_names

        for component in long_term_assets_components:
            data_row = [component] + [long_term_assets[date][component] for date in long_term_assets]
            long_term_assets_table.add_row(data_row)

        field_names = ['Component'] + [extract_date(date) for date in date_string]
        current_liabilities = {extract_date(date): {} for date in date_string}
        long_term_liabilities = {extract_date(date): {} for date in date_string}

        current_liabilities_components = ['Current Debt', 'Current Deferred Revenue', 'Pensionand Other Post Retirement Benefit Plans Current',
                                        'Other Current Liabilities', 'Accounts Payable', 'Income Tax Payable', 'Interest Payable']
        long_term_liabilities_components = ['Long Term Debt', 'Employee Benefits', 'Non Current Deferred Revenue', 
                                            'Non Current Deferred Taxes Liabilities', 'Tradeand Other Payables Non Current', 
                                            'Other Non Current Liabilities']

        for date in date_string:
            for component in current_liabilities_components:
                try:
                    current_liabilities[extract_date(date)][component] = format_large_number(balance_sheet[pd.Timestamp(date)][component])
                except KeyError:
                    current_liabilities[extract_date(date)][component] = None

            for component in long_term_liabilities_components:
                try:
                    long_term_liabilities[extract_date(date)][component] = format_large_number(balance_sheet[pd.Timestamp(date)][component])
                except KeyError:
                    long_term_liabilities[extract_date(date)][component] = None

        current_liabilities_table = PrettyTable()
        current_liabilities_table.field_names = field_names

        for component in current_liabilities_components:
            data_row = [component] + [current_liabilities[date][component] for date in current_liabilities]
            current_liabilities_table.add_row(data_row)

        long_term_liabilities_table = PrettyTable()
        long_term_liabilities_table.field_names = field_names

        for component in long_term_liabilities_components:
            data_row = [component] + [long_term_liabilities[date][component] for date in long_term_liabilities]
            long_term_liabilities_table.add_row(data_row)

        stockholders_equity = {extract_date(date): {} for date in date_string}
        stockholders_equity_components = ['Retained Earnings', 'Common Stock', 'Treasury Stock', 'Other Equity Adjustments']

        for date in stockholders_equity:
            for component in stockholders_equity_components:
                try:
                    stockholders_equity[date][component] = format_large_number(balance_sheet[pd.Timestamp(date)][component])
                except KeyError:
                    stockholders_equity[date][component] = None

        stockholders_equity_table = PrettyTable()
        field_names = ['Component'] + [extract_date(date) for date in date_string]
        stockholders_equity_table.field_names = field_names

        for component in stockholders_equity_components:
            data_row = [component] + [stockholders_equity[extract_date(date)].get(component, '') for date in date_string]
            stockholders_equity_table.add_row(data_row)

        return render(request, 'stocks/financials-balance-sheet.html', {'period': period,
                                                                        'main_components_table': main_components_table,
                                                                        'current_assets_table': current_assets_table,
                                                                        'long_term_assets_table': long_term_assets_table,
                                                                        'current_liabilities_table': current_liabilities_table,
                                                                        'long_term_liabilities_table': long_term_liabilities_table,
                                                                        'stockholders_equity_table': stockholders_equity_table})
    elif request.method == 'POST':
        action = request.POST.get('action')
    
        balance_sheet = None
        period = None

        if action == 'annual':
            balance_sheet = stock.balance_sheet
            period = 'Annual'
        elif action == 'quarterly':
            balance_sheet = stock.quarterly_balance_sheet
            period = 'Quarterly'
        else:
            return HttpResponseBadRequest('Invalid action')
        
        date_string = list(balance_sheet.keys())

        field_names = ['Component'] + [extract_date(date) for date in date_string]
        main_components = {extract_date(date): {} for date in date_string}
        components = ['Total Capitalization', 'Common Stock Equity', 'Capital Lease Obligations', 'Net Tangible Assets', 'Working Capital',
                        'Invested Capital', 'Tangible Book Value', 'Total Debt', 'Net Debt', 'Share Issued', 'Ordinary Shares Number', 
                        'Treasury Shares Number']
        
        for date in date_string:
            for component in components:
                try:
                    main_components[extract_date(date)][component] = format_large_number(balance_sheet[pd.Timestamp(date)][component])
                except KeyError:
                    main_components[extract_date(date)][component] = None

        main_components_table = PrettyTable()
        main_components_table.field_names = field_names

        for component in components:
            data_row = [component] + [main_components[date][component] for date in main_components]
            main_components_table.add_row(data_row)

        field_names = ['Component'] + [extract_date(date) for date in date_string]
        current_assets = {extract_date(date): {} for date in date_string}
        long_term_assets = {extract_date(date): {} for date in date_string}

        current_assets_components = ['Cash And Cash Equivalents', 'Other Short Term Investments', 'Accounts Receivable', 
                                    'Inventory', 'Hedging Assets Current', 'Other Current Assets']
        long_term_assets_components = ['Net PPE', 'Goodwill', 'Other Intangible Assets', 'Investments And Advances', 'Other Non Current Assets']

        for date in date_string:
            for component in current_assets_components:
                try:
                    current_assets[extract_date(date)][component] = format_large_number(balance_sheet[pd.Timestamp(date)][component])
                except KeyError:
                    current_assets[extract_date(date)][component] = None
            
            for component in long_term_assets_components:
                try:
                    long_term_assets[extract_date(date)][component] = format_large_number(balance_sheet[pd.Timestamp(date)][component])
                except KeyError:
                    long_term_assets[extract_date(date)][component] = None

        current_assets_table = PrettyTable()
        current_assets_table.field_names = field_names

        for component in current_assets_components:
            data_row = [component] + [current_assets[date][component] for date in current_assets]
            current_assets_table.add_row(data_row)

        long_term_assets_table = PrettyTable()
        long_term_assets_table.field_names = field_names

        for component in long_term_assets_components:
            data_row = [component] + [long_term_assets[date][component] for date in long_term_assets]
            long_term_assets_table.add_row(data_row)

        field_names = ['Component'] + [extract_date(date) for date in date_string]
        current_liabilities = {extract_date(date): {} for date in date_string}
        long_term_liabilities = {extract_date(date): {} for date in date_string}

        current_liabilities_components = ['Current Debt', 'Current Deferred Revenue', 'Pensionand Other Post Retirement Benefit Plans Current',
                                        'Other Current Liabilities', 'Accounts Payable', 'Income Tax Payable', 'Interest Payable']
        long_term_liabilities_components = ['Long Term Debt', 'Employee Benefits', 'Non Current Deferred Revenue', 
                                            'Non Current Deferred Taxes Liabilities', 'Tradeand Other Payables Non Current', 
                                            'Other Non Current Liabilities']

        for date in date_string:
            for component in current_liabilities_components:
                try:
                    current_liabilities[extract_date(date)][component] = format_large_number(balance_sheet[pd.Timestamp(date)][component])
                except KeyError:
                    current_liabilities[extract_date(date)][component] = None

            for component in long_term_liabilities_components:
                try:
                    long_term_liabilities[extract_date(date)][component] = format_large_number(balance_sheet[pd.Timestamp(date)][component])
                except KeyError:
                    long_term_liabilities[extract_date(date)][component] = None

        current_liabilities_table = PrettyTable()
        current_liabilities_table.field_names = field_names

        for component in current_liabilities_components:
            data_row = [component] + [current_liabilities[date][component] for date in current_liabilities]
            current_liabilities_table.add_row(data_row)

        long_term_liabilities_table = PrettyTable()
        long_term_liabilities_table.field_names = field_names

        for component in long_term_liabilities_components:
            data_row = [component] + [long_term_liabilities[date][component] for date in long_term_liabilities]
            long_term_liabilities_table.add_row(data_row)

        stockholders_equity = {extract_date(date): {} for date in date_string}
        stockholders_equity_components = ['Retained Earnings', 'Common Stock', 'Treasury Stock', 'Other Equity Adjustments']

        for date in stockholders_equity:
            for component in stockholders_equity_components:
                try:
                    stockholders_equity[date][component] = format_large_number(balance_sheet[pd.Timestamp(date)][component])
                except KeyError:
                    stockholders_equity[date][component] = None

        stockholders_equity_table = PrettyTable()
        field_names = ['Component'] + [extract_date(date) for date in date_string]
        stockholders_equity_table.field_names = field_names

        for component in stockholders_equity_components:
            data_row = [component] + [stockholders_equity[extract_date(date)].get(component, '') for date in date_string]
            stockholders_equity_table.add_row(data_row)

        return render(request, 'stocks/financials-balance-sheet.html', {'period': period,
                                                                        'main_components_table': main_components_table,
                                                                        'current_assets_table': current_assets_table,
                                                                        'long_term_assets_table': long_term_assets_table,
                                                                        'current_liabilities_table': current_liabilities_table,
                                                                        'long_term_liabilities_table': long_term_liabilities_table,
                                                                        'stockholders_equity_table': stockholders_equity_table})
    
    return render(request, 'stocks/financials-balance-sheet.html', {})

def financials_income_statement(request):
    symbol = request.session.get('symbol', 'N/A')
    stock = yf.Ticker(symbol)
    
    if request.method == 'GET':
        action = request.GET.get('action', 'annual')
      
        income_statement = stock.income_stmt
        period = 'Annual'

        date_string = list(income_statement.keys())
        
        field_names = ['Component'] + [extract_date(date) for date in date_string]
        main_components = {extract_date(date): {} for date in date_string}
        components = ['Basic EPS', 'Diluted EPS', 'Basic Average Shares', 'Diluted Average Shares', 'EBIT', 'EBITDA','Normalized EBITDA', 
                    'Reconciled Cost Of Revenue', 'Reconciled Depreciation', 'Tax Rate For Calcs', 'Tax Effect Of Unusual Items']

        for date in date_string:
            for component in components:
                try:
                    main_components[extract_date(date)][component] = format_large_number(income_statement[pd.Timestamp(date)][component])
                except KeyError:
                    main_components[extract_date(date)][component] = None

        main_components_table = PrettyTable()
        main_components_table.field_names = field_names

        for component in components:
            data_row = [component] + [main_components[date][component] for date in main_components]
            main_components_table.add_row(data_row)

       
        field_names = ['Component'] + [extract_date(date) for date in date_string]
        revenue_yearly = {extract_date(date): {} for date in date_string}
        revenue_components = ['Total Revenue', 'Operating Revenue', 'Cost Of Revenue', 'Gross Profit']

        for date in date_string:
            for component in revenue_components:
                try:
                    revenue_yearly[extract_date(date)][component] = format_large_number(income_statement[date][component])
                except KeyError:
                    revenue_yearly[extract_date(date)][component] = None

        revenue_table = PrettyTable()
        revenue_table.field_names = field_names

        for component in revenue_components:
            data_row = [component] + [revenue_yearly[date][component] for date in revenue_yearly]
            revenue_table.add_row(data_row)

        field_names = ['Component'] + [extract_date(date) for date in date_string]
        expenses_yearly = {extract_date(date): {} for date in date_string}
        expense_components = ['Total Expenses', 'Operating Expense', 'Cost Of Revenue', 'Research And Development', 'Selling And Marketing Expense', 
                            'Selling General And Administration', 'Tax Provision']

        for date in date_string:
            for component in expense_components:
                try:
                    expenses_yearly[extract_date(date)][component] = format_large_number(income_statement[pd.Timestamp(date)][component])
                except KeyError:
                    expenses_yearly[extract_date(date)][component] = None

        expenses_table = PrettyTable()
        expenses_table.field_names = field_names

        for component in expense_components:
            data_row = [component] + [expenses_yearly[date][component] for date in expenses_yearly]
            expenses_table.add_row(data_row)

        field_names = ['Component'] + [extract_date(date) for date in date_string]
        income_yearly = {extract_date(date): {} for date in date_string}
        income_components = ['Net Income', 'Net Income Common Stockholders', 'Diluted NI Availto Com Stockholders', 'Operating Income', 
                            'Pretax Income','Gross Profit', 'Interest Income']

        for date in date_string:
            for component in income_components:
                try:
                    income_yearly[extract_date(date)][component] = format_large_number(income_statement[pd.Timestamp(date)][component])
                except KeyError:
                    income_yearly[extract_date(date)][component] = None

        income_table = PrettyTable()
        income_table.field_names = field_names

        for component in income_components:
            data_row = [component] + [income_yearly[date][component] for date in income_yearly]
            income_table.add_row(data_row)

        return render(request, 'stocks/financials-income-statement.html',   {'period': period,
                                                                            'main_components_table': main_components_table,
                                                                            'revenue_table': revenue_table,
                                                                            'expenses_table': expenses_table,
                                                                            'income_table': income_table})
    elif request.method == 'POST':
        action = request.POST.get('action')
      
        income_statement = None
        period = None

        if action == 'annual':
            income_statement = stock.income_stmt
            period = 'Annual'
        elif action == 'quarterly':
            income_statement = stock.quarterly_income_stmt
            period = 'Quarterly'
        else:
            return HttpResponseBadRequest('Invalid action')
        
        date_string = list(income_statement.keys())
        
        field_names = ['Component'] + [extract_date(date) for date in date_string]
        main_components = {extract_date(date): {} for date in date_string}
        components = ['Basic EPS', 'Diluted EPS', 'Basic Average Shares', 'Diluted Average Shares', 'EBIT', 'EBITDA','Normalized EBITDA', 
                    'Reconciled Cost Of Revenue', 'Reconciled Depreciation', 'Tax Rate For Calcs', 'Tax Effect Of Unusual Items']

        for date in date_string:
            for component in components:
                try:
                    main_components[extract_date(date)][component] = format_large_number(income_statement[pd.Timestamp(date)][component])
                except KeyError:
                    main_components[extract_date(date)][component] = None

        main_components_table = PrettyTable()
        main_components_table.field_names = field_names

        for component in components:
            data_row = [component] + [main_components[date][component] for date in main_components]
            main_components_table.add_row(data_row)

       
        field_names = ['Component'] + [extract_date(date) for date in date_string]
        revenue_yearly = {extract_date(date): {} for date in date_string}
        revenue_components = ['Total Revenue', 'Operating Revenue', 'Cost Of Revenue', 'Gross Profit']

        for date in date_string:
            for component in revenue_components:
                try:
                    revenue_yearly[extract_date(date)][component] = format_large_number(income_statement[date][component])
                except KeyError:
                    revenue_yearly[extract_date(date)][component] = None

        revenue_table = PrettyTable()
        revenue_table.field_names = field_names

        for component in revenue_components:
            data_row = [component] + [revenue_yearly[date][component] for date in revenue_yearly]
            revenue_table.add_row(data_row)

        field_names = ['Component'] + [extract_date(date) for date in date_string]
        expenses_yearly = {extract_date(date): {} for date in date_string}
        expense_components = ['Total Expenses', 'Operating Expense', 'Cost Of Revenue', 'Research And Development', 'Selling And Marketing Expense', 
                            'Selling General And Administration', 'Tax Provision']

        for date in date_string:
            for component in expense_components:
                try:
                    expenses_yearly[extract_date(date)][component] = format_large_number(income_statement[pd.Timestamp(date)][component])
                except KeyError:
                    expenses_yearly[extract_date(date)][component] = None

        expenses_table = PrettyTable()
        expenses_table.field_names = field_names

        for component in expense_components:
            data_row = [component] + [expenses_yearly[date][component] for date in expenses_yearly]
            expenses_table.add_row(data_row)

        field_names = ['Component'] + [extract_date(date) for date in date_string]
        income_yearly = {extract_date(date): {} for date in date_string}
        income_components = ['Net Income', 'Net Income Common Stockholders', 'Diluted NI Availto Com Stockholders', 'Operating Income', 
                            'Pretax Income','Gross Profit', 'Interest Income']

        for date in date_string:
            for component in income_components:
                try:
                    income_yearly[extract_date(date)][component] = format_large_number(income_statement[pd.Timestamp(date)][component])
                except KeyError:
                    income_yearly[extract_date(date)][component] = None

        income_table = PrettyTable()
        income_table.field_names = field_names

        for component in income_components:
            data_row = [component] + [income_yearly[date][component] for date in income_yearly]
            income_table.add_row(data_row)

        return render(request, 'stocks/financials-income-statement.html',   {'period': period,
                                                                            'main_components_table': main_components_table,
                                                                            'revenue_table': revenue_table,
                                                                            'expenses_table': expenses_table,
                                                                            'income_table': income_table})
    return render(request, 'stocks/financials-income-statement.html', {})

def financials_cash_flow(request):
    symbol = request.session.get('symbol', 'N/A')
    stock = yf.Ticker(symbol)
    
    if request.method == 'GET':
        action = request.GET.get('action', 'annual')

        cashflow = stock.cashflow
        period = 'Annual'

        date_string = list(cashflow.keys())

        field_names = ['Component'] + [extract_date(date) for date in date_string]
        main_components = {extract_date(date): {} for date in date_string}
        components = ['Capital Expenditure', 'Issuance Of Capital Stock', 'Issuance Of Debt', 'Repayment Of Debt', 
                        'Repurchase Of Capital Stock', 'Income Tax Paid Supplemental Data', 'Interest Paid Supplemental Data', 'Free Cash Flow']

        for date in date_string:
            for component in components:
                try:
                    main_components[extract_date(date)][component] = format_large_number(cashflow[pd.Timestamp(date)][component])
                except KeyError:
                    main_components[extract_date(date)][component] = None

        main_components_table = PrettyTable()
        main_components_table.field_names = field_names

        for component in components:
            data_row = [component] + [main_components[date][component] for date in main_components]
            main_components_table.add_row(data_row)

        field_names = ['Component'] + [extract_date(date) for date in date_string]
        cfo_yearly = {extract_date(date): {} for date in date_string}
        cfo_components = ['Operating Cash Flow', 'Change In Working Capital', 'Change In Other Working Capital', 'Stock Based Compensation', 
                        'Depreciation And Amortization', 'Depreciation Amortization Depletion', 'Deferred Tax', 'Net Income From Continuing Operations', 'Changes In Account Receivables']

        for date in date_string:
            for component in cfo_components:
                try:
                    cfo_yearly[extract_date(date)][component] = format_large_number(cashflow[pd.Timestamp(date)][component])
                except KeyError:
                    cfo_yearly[extract_date(date)][component] = None

        cfo_table = PrettyTable()
        cfo_table.field_names = field_names

        for component in cfo_components:
            data_row = [component] + [cfo_yearly[date][component] for date in cfo_yearly]
            cfo_table.add_row(data_row)

        field_names = ['Component'] + [extract_date(date) for date in date_string]
        cfi_yearly = {extract_date(date): {} for date in date_string}
        cfi_components = ['Net PPE Purchase And Sale', 'Net Business Purchase And Sale', 'Net Investment Purchase And Sale', 'Purchase Of Investment', 'Sale Of Investment']

        for date in date_string:
            for component in cfi_components:
                try:
                    cfi_yearly[extract_date(date)][component] = format_large_number(cashflow[pd.Timestamp(date)][component])
                except KeyError:
                    cfi_yearly[extract_date(date)][component] = None

        cfi_table = PrettyTable()
        cfi_table.field_names = field_names

        for component in cfi_components:
            data_row = [component] + [cfi_yearly[date][component] for date in cfi_yearly]
            cfi_table.add_row(data_row)

        field_names = ['Component'] + [extract_date(date) for date in date_string]
        cff_yearly = {extract_date(date): {} for date in date_string}
        cff_components = ['Net Long Term Debt Issuance', 'Long Term Debt Issuance', 'Long Term Debt Payments', 'Net Common Stock Issuance', 'Common Stock Issuance', 
                        'Common Stock Payments', 'Net Other Financing Charges']

        for date in date_string:
            for component in cff_components:
                try:
                    cff_yearly[extract_date(date)][component] = format_large_number(cashflow[pd.Timestamp(date)][component])
                except KeyError:
                    cff_yearly[extract_date(date)][component] = None

        cff_table = PrettyTable()
        cff_table.field_names = field_names

        for component in cff_components:
            data_row = [component] + [cff_yearly[date][component] for date in cff_yearly]
            cff_table.add_row(data_row)

        field_names = ['Component'] + [extract_date(date) for date in date_string]
        ecp_yearly = {extract_date(date): {} for date in date_string}
        ecp_components = ['Changes In Cash', 'Effect Of Exchange Rate Changes', 'Beginning Cash Position']

        for date in date_string:
            for component in ecp_components:
                try:
                    ecp_yearly[extract_date(date)][component] = format_large_number(cashflow[pd.Timestamp(date)][component])
                except KeyError:
                    ecp_yearly[extract_date(date)][component] = None

        ecp_table = PrettyTable()
        ecp_table.field_names = field_names

        for component in ecp_components:
            data_row = [component] + [ecp_yearly[date][component] for date in ecp_yearly]
            ecp_table.add_row(data_row)

        return render(request, 'stocks/financials-cash-flow.html', {'period': period,
                                                                    'main_components_table': main_components_table,
                                                                    'cfo_table': cfo_table,
                                                                    'cfi_table': cfi_table,
                                                                    'cff_table': cff_table,
                                                                    'ecp_table': ecp_table})
    elif request.method == 'POST':
        action = request.POST.get('action')
        
        cashflow = None
        period = None

        if action == 'annual':
            cashflow = stock.cashflow
            period = 'Annual'
        elif action == 'quarterly':
            cashflow = stock.quarterly_cashflow
            period = 'Quarterly'
        else:
            return HttpResponseBadRequest('Invalid action')
        
        date_string = list(cashflow.keys())

        field_names = ['Component'] + [extract_date(date) for date in date_string]
        main_components = {extract_date(date): {} for date in date_string}
        components = ['Capital Expenditure', 'Issuance Of Capital Stock', 'Issuance Of Debt', 'Repayment Of Debt', 
                        'Repurchase Of Capital Stock', 'Income Tax Paid Supplemental Data', 'Interest Paid Supplemental Data', 'Free Cash Flow']

        for date in date_string:
            for component in components:
                try:
                    main_components[extract_date(date)][component] = format_large_number(cashflow[pd.Timestamp(date)][component])
                except KeyError:
                    main_components[extract_date(date)][component] = None

        main_components_table = PrettyTable()
        main_components_table.field_names = field_names

        for component in components:
            data_row = [component] + [main_components[date][component] for date in main_components]
            main_components_table.add_row(data_row)

        field_names = ['Component'] + [extract_date(date) for date in date_string]
        cfo_yearly = {extract_date(date): {} for date in date_string}
        cfo_components = ['Operating Cash Flow', 'Change In Working Capital', 'Change In Other Working Capital', 'Stock Based Compensation', 
                        'Depreciation And Amortization', 'Depreciation Amortization Depletion', 'Deferred Tax', 'Net Income From Continuing Operations', 'Changes In Account Receivables']

        for date in date_string:
            for component in cfo_components:
                try:
                    cfo_yearly[extract_date(date)][component] = format_large_number(cashflow[pd.Timestamp(date)][component])
                except KeyError:
                    cfo_yearly[extract_date(date)][component] = None

        cfo_table = PrettyTable()
        cfo_table.field_names = field_names

        for component in cfo_components:
            data_row = [component] + [cfo_yearly[date][component] for date in cfo_yearly]
            cfo_table.add_row(data_row)

        field_names = ['Component'] + [extract_date(date) for date in date_string]
        cfi_yearly = {extract_date(date): {} for date in date_string}
        cfi_components = ['Net PPE Purchase And Sale', 'Net Business Purchase And Sale', 'Net Investment Purchase And Sale', 'Purchase Of Investment', 'Sale Of Investment']

        for date in date_string:
            for component in cfi_components:
                try:
                    cfi_yearly[extract_date(date)][component] = format_large_number(cashflow[pd.Timestamp(date)][component])
                except KeyError:
                    cfi_yearly[extract_date(date)][component] = None

        cfi_table = PrettyTable()
        cfi_table.field_names = field_names

        for component in cfi_components:
            data_row = [component] + [cfi_yearly[date][component] for date in cfi_yearly]
            cfi_table.add_row(data_row)

        field_names = ['Component'] + [extract_date(date) for date in date_string]
        cff_yearly = {extract_date(date): {} for date in date_string}
        cff_components = ['Net Long Term Debt Issuance', 'Long Term Debt Issuance', 'Long Term Debt Payments', 'Net Common Stock Issuance', 'Common Stock Issuance', 
                        'Common Stock Payments', 'Net Other Financing Charges']

        for date in date_string:
            for component in cff_components:
                try:
                    cff_yearly[extract_date(date)][component] = format_large_number(cashflow[pd.Timestamp(date)][component])
                except KeyError:
                    cff_yearly[extract_date(date)][component] = None

        cff_table = PrettyTable()
        cff_table.field_names = field_names

        for component in cff_components:
            data_row = [component] + [cff_yearly[date][component] for date in cff_yearly]
            cff_table.add_row(data_row)

        field_names = ['Component'] + [extract_date(date) for date in date_string]
        ecp_yearly = {extract_date(date): {} for date in date_string}
        ecp_components = ['Changes In Cash', 'Effect Of Exchange Rate Changes', 'Beginning Cash Position']

        for date in date_string:
            for component in ecp_components:
                try:
                    ecp_yearly[extract_date(date)][component] = format_large_number(cashflow[pd.Timestamp(date)][component])
                except KeyError:
                    ecp_yearly[extract_date(date)][component] = None

        ecp_table = PrettyTable()
        ecp_table.field_names = field_names

        for component in ecp_components:
            data_row = [component] + [ecp_yearly[date][component] for date in ecp_yearly]
            ecp_table.add_row(data_row)

        return render(request, 'stocks/financials-cash-flow.html', {'period': period,
                                                                    'main_components_table': main_components_table,
                                                                    'cfo_table': cfo_table,
                                                                    'cfi_table': cfi_table,
                                                                    'cff_table': cff_table,
                                                                    'ecp_table': ecp_table})
    return render(request, 'stocks/financials-cash-flow.html', {})
     
def holders(request):
    symbol = request.session.get('symbol', 'N/A')
   
    holders = get_holders(symbol)
    
    major_holders_table = None
    direct_holders_table = None
    institutional_holders_table = None
    
    try:
        major_holders = holders['Major Holders']
        if not major_holders.empty:
            major_holders_df = pd.DataFrame(major_holders)
            major_holders_df.columns = ['Breakdown', 'Description']

            major_holders_df['Description'] = major_holders_df['Description'].str.replace('%', 'Percentage')

            major_holders_table = PrettyTable()
            major_holders_table.field_names = ['Breakdown', 'Description']

            for row in major_holders_df.itertuples(index=False):
                major_holders_table.add_row(row)
    except KeyError:
        pass    
    
    try:
        direct_holders = holders['Direct Holders (Forms 3 and 4)']
        if not direct_holders.empty:
            direct_holders_table = PrettyTable()
            direct_holders_table.field_names = ["Holder", "Shares Held", "Percentage Held", "Value"]

            for index, row in direct_holders.iterrows():
                holder = row['Holder']
                shares_held = row['Shares']
                percentage = row['% Out']
                value = format_large_number(row['Value'])
                direct_holders_table.add_row([holder, shares_held, percentage, value]) 
    except KeyError:
        pass  

    try:
        institutional_holders = holders['Top Institutional Holders']
        if not institutional_holders.empty:
            institutional_holders_table = PrettyTable()
            institutional_holders_table.field_names = ["Holder", "Shares Held", "Percentage Held", "Value"]

            for index, row in institutional_holders.iterrows():
                holder = row['Holder']
                shares_held = row['Shares']
                percentage = row['% Out']
                value = format_large_number(row['Value'])
                institutional_holders_table.add_row([holder, shares_held, percentage, value])  
    except KeyError:
        pass  

    return render(request, 'stocks/holders.html', {'major_holders_table': major_holders_table, 
                                                   'direct_holders_table': direct_holders_table, 
                                                   'institutional_holders_table': institutional_holders_table})

def peers(request):
    symbol = request.session.get('symbol', 'N/A')

    combined_tickers = get_top_n_peers(symbol, 8)
 
    peer_table = get_peer_table(symbol, combined_tickers)
    valuation_metrics_table = get_valuation_metrics_table(symbol, combined_tickers)
    profitability_metrics_table = get_profitability_metrics_table(symbol, combined_tickers)
    leverage_metrics_table = get_leverage_metrics_table(symbol, combined_tickers)
    risk_metrics_table = get_risk_metrics_table(symbol, combined_tickers)
    growth_rates_table = get_expected_annual_earnings_growth_table(symbol, combined_tickers)
    
    return render(request, 'stocks/peers.html', {'peer_table': peer_table,
                                                 'valuation_metrics_table': valuation_metrics_table,
                                                 'profitability_metrics_table': profitability_metrics_table,
                                                 'leverage_metrics_table': leverage_metrics_table,
                                                 'risk_metrics_table': risk_metrics_table,
                                                 'growth_rates_table': growth_rates_table})

def indices(request):
    try:
        choice = request.session.get('symbol', 'N/A')
        stock = yf.Ticker(choice)

        max_attempts = 3
        attempt = 1

        while attempt <= max_attempts:
            try:
                df_list = pd.read_html('https://finance.yahoo.com/world-indices/')
                majorStockIdx = df_list[0]
                break  
            except Exception as e:
                print(f"Attempt {attempt}: An error occurred while fetching data: {e}")
                if attempt == max_attempts:
                    print(f"Reached maximum number of attempts ({max_attempts}). Exiting.")
                    majorStockIdx = None
                    break  
                else:
                    print("Retrying...")
                    attempt += 1

        timezone = stock.info.get('timeZoneFullName')
        stock_country = timezone.split('/')[0].strip()

        symbols_by_country = {}

        for index, row in majorStockIdx.iterrows():
            symbol = row['Symbol']
            ticker = yf.Ticker(symbol)
            
            timezone = ticker.info.get('timeZoneFullName')
            country = timezone.split('/')[0].strip()
            
            if country == "Australia" or country == "Pacific":
                country = "Australia/Pacific"

            symbols_by_country.setdefault(country, []).append(symbol)

        stock_country_symbols = symbols_by_country.get(stock_country, [])
        stock_country_symbols = get_top_n_indices(stock_country_symbols, 8, 'max')
        indices_table = get_index_info(stock_country_symbols)

        choice_metrics = calculate_performance_metrics(choice)

        performance_metrics_table = PrettyTable()
        performance_metrics_table.field_names = ['Name', 'Total Return', 'Annualized Return', 'Volatility', 'Maximum Drawdown']

        if choice_metrics:
            performance_metrics_table.add_row([get_ticker_name(choice), choice_metrics['Total Return'], choice_metrics['Annualized Return'], 
                                choice_metrics['Volatility'], choice_metrics['Maximum Drawdown']])

        for symbol in stock_country_symbols:
            index_metrics = calculate_performance_metrics(symbol)
            if index_metrics:
                performance_metrics_table.add_row([get_ticker_name(symbol), index_metrics['Total Return'], index_metrics['Annualized Return'], 
                                    index_metrics['Volatility'], index_metrics['Maximum Drawdown']])
            else:
                print(f"No metrics available for {symbol}")
                stock_country_symbols.remove(symbol)

        market_metrics_table = PrettyTable()
        market_metrics_table.field_names = ['Index', 'Market Capitalization', 'Beta', 'Correlation', 'Liquidity (Avg. Daily Volume)']

        for symbol in stock_country_symbols:
            market_metrics = calculate_market_metrics(choice, symbol)

            if market_metrics:
                market_metrics_table.add_row([
                    get_ticker_name(symbol),
                    market_metrics.get('Market Capitalization', 'N/A'),
                    market_metrics.get('Beta', 'N/A'),
                    market_metrics.get('Correlation', 'N/A'),
                    market_metrics.get('Liquidity (Avg. Daily Volume)', 'N/A')
                ])

        sma_table = PrettyTable()
        sma_table.field_names = ['Name', 'SMA 50', 'SMA 200', '50-Day Average', '200-Day Average']

        choice_sma_50 = calculate_sma(choice, window=50)
        choice_sma_200 = calculate_sma(choice, window=200)

        if choice_sma_50 is not None and choice_sma_200 is not None:
            sma_table.add_row([get_ticker_name(choice), format_large_number("{:.2f}".format(choice_sma_50.iloc[-1])), 
                            format_large_number("{:.2f}".format(choice_sma_200.iloc[-1])), get_fifty_day_average(choice), get_two_hundred_day_average(choice)])

        for symbol in stock_country_symbols:
            sma_50 = calculate_sma(symbol, window=50)
            sma_200 = calculate_sma(symbol, window=200)

            if sma_50 is not None and sma_200 is not None:
                sma_table.add_row([get_ticker_name(symbol), format_large_number("{:.2f}".format(sma_50.iloc[-1])), 
                                format_large_number("{:.2f}".format(sma_200.iloc[-1])), get_fifty_day_average(symbol), get_two_hundred_day_average(symbol)])

        return render(request, 'stocks/indices.html', {'indices_table': indices_table, 
                                                'performance_metrics_table': performance_metrics_table, 
                                                'market_metrics_table': market_metrics_table, 
                                                'sma_table': sma_table})
    except Exception as e:
        return HttpResponseServerError(f"An error occurred: {e}")

def clear_session(request):
    request.session.flush()  
    return redirect('index')
