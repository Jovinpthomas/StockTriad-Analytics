# StockTriad-Analytics
StockTriad-Analytics is a Django web application for comprehensive stock analysis. It offers dynamic filtering, detailed financial statements, risk assessments, peer comparisons, and interactive visualizations. The app uses real-time market data and sentiment analysis to help users make informed investment decisions.

## Features
- Intuitive user interface for easy navigation
- Dynamic filtering options for customized analysis
- Real-time data updates for the latest market information
- Detailed financial statements, risk assessments, and peer comparisons
- Interactive visualizations using Plotly
- Support for sentiment analysis and news scraping

## Installation
1. Clone the Repository:
   ```bash
   git clone https://github.com/Jovinpthomas/StockTriad-Analytics.git
   ```
2. Navigate to the Project Directory:
   ```bash
   cd StockTriad-Analytics
   ```
3. Create a Virtual Environment:
   ```bash
   python -m venv env
   ```
4. Activate the Virtual Environment:
   ```bash
   .\env\Scripts\activate
   ```
5. Install Dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   ```bash
   python download_nltk_data.py
   ```
6. Run the Application:
   ```bash
   python manage.py runserver
   ```

## Setting Up Alpha Vantage and Polygon APIs
- Alpha Vantage: [Sign up here](https://www.alphavantage.co/support/#api-key) to get your free API key.
- Polygon: [Sign up here](https://polygon.io/dashboard/login) to get your API key.
Paste your API Keys in [views.py](stock_analysis_project/stocks/views.py)
   
## Additional Resources
- [Structure.txt](Structure.txt): Structure of the website.
- [StockTriad Analytics.pdf](StockTriad%20Analytics.pdf): Detailed documentation of the project's analysis and findings.
- [StockTriad Analytics.ipynb](stocktriad-analytics.ipynb): Jupyter Notebook containing the project code and analysis.
