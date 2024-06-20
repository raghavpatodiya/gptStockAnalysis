import os
import json
import openai
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
import numpy as np
from sklearn.linear_model import LinearRegression

# LangChain imports
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")     

PERSIST = False

if PERSIST and os.path.exists("persist"):
    print("Reusing index...\n")
    vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
    index = VectorStoreIndexWrapper(vectorstore=vectorstore)
else:
    loader = DirectoryLoader("data/")
    if PERSIST:
        index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory": "persist"}).from_loaders([loader])
    else:
        index = VectorstoreIndexCreator(embedding=OpenAIEmbeddings()).from_loaders([loader])

retrieval_chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(model="gpt-3.5-turbo"),
    retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
)

def get_stock_price(ticker):
    return str(yf.Ticker(ticker).history(period='1y').iloc[-1].Close)

def calculate_SMA(ticker, window):
    data = yf.Ticker(ticker).history(period='1y').Close
    return str(data.rolling(window=window).mean().iloc[-1])

def calculate_EMA(ticker, window):
    data = yf.Ticker(ticker).history(period='1y').Close
    return str(data.ewm(span=window, adjust=False).mean().iloc[-1])

def calculate_RSI(ticker):
    data = yf.Ticker(ticker).history(period='1y').Close
    delta = data.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=14-1, adjust=False).mean()
    ema_down = down.ewm(com=14-1, adjust=False).mean()
    rs = ema_up / ema_down
    return str(100 - (100 / (1 + rs)).iloc[-1])

def calculate_MACD(ticker):
    data = yf.Ticker(ticker).history(period='1y').Close
    short_EMA = data.ewm(span=12, adjust=False).mean()
    long_EMA = data.ewm(span=26, adjust=False).mean()
    MACD = short_EMA - long_EMA
    signal = MACD.ewm(span=9, adjust=False).mean()
    MACD_histogram = MACD - signal
    return f'{MACD[-1]}, {signal[-1]}, {MACD_histogram[-1]}'

def plot_stock_price(ticker):
    data = yf.Ticker(ticker).history(period='1y')
    plt.figure(figsize=(10, 5))
    plt.plot(data.index, data.Close)
    plt.title(f'{ticker} Stock Price Over Last Year')
    plt.xlabel('Date')
    plt.ylabel('Stock Price ($)')
    plt.grid(True)
    plt.savefig('stock.png')
    plt.close()

def calculate_bollinger_bands(ticker, window, num_std_dev):
    data = yf.Ticker(ticker).history(period='1y').Close
    sma = data.rolling(window=window).mean()
    std_dev = data.rolling(window=window).std()
    upper_band = sma + (std_dev * num_std_dev)
    lower_band = sma - (std_dev * num_std_dev)
    return {
        'sma': sma.iloc[-1],
        'upper_band': upper_band.iloc[-1],
        'lower_band': lower_band.iloc[-1]
    }

def calculate_stochastic_oscillator(ticker, window):
    data = yf.Ticker(ticker).history(period='1y')
    low_min = data['Low'].rolling(window=window).min()
    high_max = data['High'].rolling(window=window).max()
    stochastic = ((data['Close'] - low_min) / (high_max - low_min)) * 100
    return stochastic.iloc[-1]

def calculate_atr(ticker, window):
    data = yf.Ticker(ticker).history(period='1y')
    high_low = data['High'] - data['Low']
    high_close = abs(data['High'] - data['Close'].shift())
    low_close = abs(data['Low'] - data['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(window=window).mean()
    return atr.iloc[-1]

def calculate_obv(ticker):
    data = yf.Ticker(ticker).history(period='1y')
    obv = (data['Volume'] * ((data['Close'] - data['Open']) / data['Open'])).cumsum()
    return obv.iloc[-1]

def calculate_macd_histogram(ticker):
    data = yf.Ticker(ticker).history(period='1y').Close
    short_ema = data.ewm(span=12, adjust=False).mean()
    long_ema = data.ewm(span=26, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=9, adjust=False).mean()
    macd_histogram = macd - signal
    return macd_histogram.iloc[-1]

def calculate_vwap(ticker):
    data = yf.Ticker(ticker).history(period='1y')
    vwap = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
    return vwap.iloc[-1]

def calculate_linear_regression_trend(ticker):
    data = yf.Ticker(ticker).history(period='1y').Close
    data = data.reset_index()
    data['Date_ordinal'] = pd.to_datetime(data['Date']).map(pd.Timestamp.toordinal)
    X = data['Date_ordinal'].values.reshape(-1, 1)
    y = data['Close'].values
    model = LinearRegression().fit(X, y)
    trend = model.predict(X)
    return trend[-1], model.coef_[0]

def calculate_adx(ticker, window=14):
    data = yf.Ticker(ticker).history(period='1y')
    high = data['High']
    low = data['Low']
    close = data['Close']
    plus_dm = high.diff().clip(lower=0)
    minus_dm = -low.diff().clip(upper=0)
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(window).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/window).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/window).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(window).mean()
    return adx.iloc[-1]

def calculate_cci(ticker, window=20):
    data = yf.Ticker(ticker).history(period='1y')
    tp = (data['High'] + data['Low'] + data['Close']) / 3
    sma = tp.rolling(window=window).mean()
    mad = (tp - sma).rolling(window=window).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    cci = (tp - sma) / (0.015 * mad)
    return cci.iloc[-1]

def calculate_beta(ticker, benchmark='^GSPC'):
    stock_data = yf.Ticker(ticker).history(period='5y')
    benchmark_data = yf.Ticker(benchmark).history(period='5y')
    stock_returns = stock_data['Close'].pct_change()[1:]
    benchmark_returns = benchmark_data['Close'].pct_change()[1:]
    beta = stock_returns.cov(benchmark_returns) / benchmark_returns.var()
    return beta

def calculate_sharpe_ratio(ticker, risk_free_rate=0.01):
    data = yf.Ticker(ticker).history(period='1y')['Close']
    returns = data.pct_change().dropna()
    excess_returns = returns - (risk_free_rate / 252)
    sharpe_ratio = excess_returns.mean() / excess_returns.std() * (252 ** 0.5)
    return sharpe_ratio

def calculate_sortino_ratio(ticker, risk_free_rate=0.01):
    data = yf.Ticker(ticker).history(period='1y')['Close']
    returns = data.pct_change().dropna()
    negative_returns = returns[returns < 0]
    downside_deviation = negative_returns.std() * (252 ** 0.5)
    annualized_return = (1 + returns.mean()) ** 252 - 1
    excess_return = annualized_return - risk_free_rate
    sortino_ratio = excess_return / downside_deviation
    return sortino_ratio

def calculate_treynor_ratio(ticker, benchmark='^GSPC', risk_free_rate=0.01):
    stock_data = yf.Ticker(ticker).history(period='1y')['Close']
    benchmark_data = yf.Ticker(benchmark).history(period='1y')['Close']
    stock_returns = stock_data.pct_change().dropna()
    benchmark_returns = benchmark_data.pct_change().dropna()
    beta = stock_returns.cov(benchmark_returns) / benchmark_returns.var()
    annualized_return = (1 + stock_returns.mean()) ** 252 - 1
    excess_return = annualized_return - risk_free_rate
    treynor_ratio = excess_return / beta
    return treynor_ratio

def calculate_calmar_ratio(ticker):
    data = yf.Ticker(ticker).history(period='3y')['Close']
    returns = data.pct_change().dropna()
    annualized_return = (1 + returns.mean()) ** 252 - 1
    max_drawdown = ((data / data.cummax()) - 1).min()
    calmar_ratio = annualized_return / abs(max_drawdown)
    return calmar_ratio

def calculate_pe_ratio(ticker):
    stock_info = yf.Ticker(ticker).info
    pe_ratio = stock_info.get('trailingPE', None)
    return pe_ratio

def calculate_pb_ratio(ticker):
    stock_info = yf.Ticker(ticker).info
    pb_ratio = stock_info.get('priceToBook', None)
    return pb_ratio

def calculate_dividend_yield(ticker):
    stock_info = yf.Ticker(ticker).info
    dividend_yield = stock_info.get('dividendYield', None)
    return dividend_yield

def calculate_eps(ticker):
    stock_info = yf.Ticker(ticker).info
    eps = stock_info.get('trailingEps', None)
    return eps

def calculate_de_ratio(ticker):
    stock_info = yf.Ticker(ticker).info
    de_ratio = stock_info.get('debtToEquity', None)
    return de_ratio

def calculate_price_to_sales(ticker):
    stock = yf.Ticker(ticker)
    price = stock.history(period='1d').iloc[-1].Close
    sales = stock.info['totalRevenue']
    shares_outstanding = stock.info['sharesOutstanding']
    return price / (sales / shares_outstanding)

def calculate_roe(ticker):
    stock = yf.Ticker(ticker)
    net_income = stock.info['netIncomeToCommon']
    equity = stock.info['totalStockholderEquity']
    return net_income / equity

def calculate_roa(ticker):
    stock = yf.Ticker(ticker)
    net_income = stock.info['netIncomeToCommon']
    total_assets = stock.info['totalAssets']
    return net_income / total_assets

def calculate_peg(ticker):
    stock = yf.Ticker(ticker)
    pe_ratio = stock.info['forwardPE']
    growth_rate = stock.info['earningsQuarterlyGrowth']
    return pe_ratio / growth_rate

def calculate_current_ratio(ticker):
    stock = yf.Ticker(ticker)
    current_assets = stock.info['totalCurrentAssets']
    current_liabilities = stock.info['totalCurrentLiabilities']
    return current_assets / current_liabilities

def calculate_quick_ratio(ticker):
    stock = yf.Ticker(ticker)
    current_assets = stock.info['totalCurrentAssets']
    inventory = stock.info['inventory']
    current_liabilities = stock.info['totalCurrentLiabilities']
    return (current_assets - inventory) / current_liabilities

def calculate_dividend_payout_ratio(ticker):
    stock = yf.Ticker(ticker)
    dividends_per_share = stock.info['dividendRate']
    eps = stock.info['trailingEps']
    return dividends_per_share / eps

def calculate_free_cash_flow(ticker):
    stock = yf.Ticker(ticker)
    operating_cash_flow = stock.info['operatingCashflow']
    capital_expenditure = stock.info['capitalExpenditures']
    return operating_cash_flow - capital_expenditure

def calculate_gross_margin(ticker):
    stock = yf.Ticker(ticker)
    gross_profit = stock.info['grossProfits']
    total_revenue = stock.info['totalRevenue']
    return gross_profit / total_revenue

def calculate_operating_margin(ticker):
    stock = yf.Ticker(ticker)
    operating_income = stock.info['operatingIncome']
    total_revenue = stock.info['totalRevenue']
    return operating_income / total_revenue

functions = [
    {
        'name': 'get_stock_price',
        'description': 'Gets the latest stock price given the ticker symbol of a company',
        'parameters': {
            'type': 'object',
            'properties': {
                'ticker': {
                    'type': 'string',
                    'description': 'The stock ticker symbol for a company (for example AAPL for Apple)'
                }
            },
            'required': ['ticker'],
        }
    },
    {
        'name': 'calculate_SMA',
        'description': 'Calculates the Simple Moving Average (SMA) for a given stock ticker and window size',
        'parameters': {
            'type': 'object',
            'properties': {
                'ticker': {
                    'type': 'string',
                    'description': 'The stock ticker symbol for a company (for example AAPL for Apple)'
                },
                'window': {
                    'type': 'integer',
                    'description': 'The window size for calculating the SMA (e.g., 20 days)'
                }
            },
            'required': ['ticker', 'window'],
        }
    },
    {
        'name': 'calculate_EMA',
        'description': 'Calculates the Exponential Moving Average (EMA) for a given stock ticker and window size',
        'parameters': {
            'type': 'object',
            'properties': {
                'ticker': {
                    'type': 'string',
                    'description': 'The stock ticker symbol for a company (for example AAPL for Apple)'
                },
                'window': {
                    'type': 'integer',
                    'description': 'The window size for calculating the EMA (e.g., 20 days)'
                }
            },
            'required': ['ticker', 'window'],
        }
    },
    {
        'name': 'calculate_RSI',
        'description': 'Calculates the Relative Strength Index (RSI) for a given stock ticker',
        'parameters': {
            'type': 'object',
            'properties': {
                'ticker': {
                    'type': 'string',
                    'description': 'The stock ticker symbol for a company (for example AAPL for Apple)'
                }
            },
            'required': ['ticker'],
        }
    },
    {
        'name': 'calculate_MACD',
        'description': 'Calculates the Moving Average Convergence Divergence (MACD) for a given stock ticker.',
        'parameters': {
            'type': 'object',
            'properties': {
                'ticker': {
                    'type': 'string',
                    'description': 'The stock ticker symbol for a company (for example AAPL for Apple)'
                }
            },
            'required': ['ticker'],
        }
    },
    {
        'name': 'plot_stock_price',
        'description': 'Plots the stock price for a given ticker over the past year.',
        'parameters': {
            'type': 'object',
            'properties': {
                'ticker': {
                    'type': 'string',
                    'description': 'The stock ticker symbol for a company (for example AAPL for Apple)'
                }
            },
            'required': ['ticker'],
        }
    },
    {
        'name': 'calculate_bollinger_bands',
        'description': 'Calculates the Bollinger Bands for a given stock ticker, window size, and number of standard deviations.',
        'parameters': {
            'type': 'object',
            'properties': {
                'ticker': {
                    'type': 'string',
                    'description': 'The stock ticker symbol for a company (for example AAPL for Apple)'
                },
                'window': {
                    'type': 'integer',
                    'description': 'The window size for calculating the Bollinger Bands (e.g., 20 days)'
                },
                'num_std_dev': {
                    'type': 'integer',
                    'description': 'The number of standard deviations for the Bollinger Bands (e.g., 2)'
                }
            },
            'required': ['ticker', 'window', 'num_std_dev'],
        }
    },
    {
        'name': 'calculate_stochastic_oscillator',
        'description': 'Calculates the Stochastic Oscillator for a given stock ticker and window size.',
        'parameters': {
            'type': 'object',
            'properties': {
                'ticker': {
                    'type': 'string',
                    'description': 'The stock ticker symbol for a company (for example AAPL for Apple)'
                },
                'window': {
                    'type': 'integer',
                    'description': 'The window size for calculating the Stochastic Oscillator (e.g., 14 days)'
                }
            },
            'required': ['ticker', 'window'],
        }
    },
    {
        'name': 'calculate_atr',
        'description': 'Calculates the Average True Range (ATR) for a given stock ticker and window size.',
        'parameters': {
            'type': 'object',
            'properties': {
                'ticker': {
                    'type': 'string',
                    'description': 'The stock ticker symbol for a company (for example AAPL for Apple)'
                },
                'window': {
                    'type': 'integer',
                    'description': 'The window size for calculating the ATR (e.g., 14 days)'
                }
            },
            'required': ['ticker', 'window'],
        }
    },
    {
        'name': 'calculate_obv',
        'description': 'Calculates the On-Balance Volume (OBV) for a given stock ticker.',
        'parameters': {
            'type': 'object',
            'properties': {
                'ticker': {
                    'type': 'string',
                    'description': 'The stock ticker symbol for a company (for example AAPL for Apple)'
                }
            },
            'required': ['ticker'],
        }
    },
    {
        'name': 'calculate_macd_histogram',
        'description': 'Calculates the MACD histogram for a given stock ticker.',
        'parameters': {
            'type': 'object',
            'properties': {
                'ticker': {
                    'type': 'string',
                    'description': 'The stock ticker symbol for a company (for example AAPL for Apple)'
                }
            },
            'required': ['ticker'],
        }
    },
    {
        'name': 'calculate_vwap',
        'description': 'Calculates the Volume Weighted Average Price (VWAP) for a given stock ticker.',
        'parameters': {
            'type': 'object',
            'properties': {
                'ticker': {
                    'type': 'string',
                    'description': 'The stock ticker symbol for a company (for example AAPL for Apple)'
                }
            },
            'required': ['ticker'],
        }
    },
    {
        'name': 'calculate_linear_regression_trend',
        'description': 'Calculates the linear regression trend for a given stock ticker.',
        'parameters': {
            'type': 'object',
            'properties': {
                'ticker': {
                    'type': 'string',
                    'description': 'The stock ticker symbol for a company (for example AAPL for Apple)'
                }
            },
            'required': ['ticker'],
        }
    },
    {
        'name': 'calculate_adx',
        'description': 'Calculates the Average Directional Index (ADX) for a given stock ticker and window size.',
        'parameters': {
            'type': 'object',
            'properties': {
                'ticker': {
                    'type': 'string',
                    'description': 'The stock ticker symbol for a company (for example AAPL for Apple)'
                },
                'window': {
                    'type': 'integer',
                    'description': 'The window size for calculating the ADX (e.g., 14 days)'
                }
            },
            'required': ['ticker', 'window'],
        }
    },
    {
        'name': 'calculate_cci',
        'description': 'Calculates the Commodity Channel Index (CCI) for a given stock ticker and window size.',
        'parameters': {
            'type': 'object',
            'properties': {
                'ticker': {
                    'type': 'string',
                    'description': 'The stock ticker symbol for a company (for example AAPL for Apple)'
                },
                'window': {
                    'type': 'integer',
                    'description': 'The window size for calculating the CCI (e.g., 20 days)'
                }
            },
            'required': ['ticker', 'window'],
        }
    },
    {
        'name': 'calculate_beta',
        'description': 'Calculates the Beta (volatility measure) of a stock relative to a benchmark index.',
        'parameters': {
            'type': 'object',
            'properties': {
                'ticker': {'type': 'string', 'description': 'The stock ticker symbol (e.g., AAPL for Apple)'},
                'benchmark': {'type': 'string', 'description': 'The benchmark index ticker symbol (default is ^GSPC for S&P 500)'}
            },
            'required': ['ticker']
        }
    },
    {
        'name': 'calculate_sharpe_ratio',
        'description': 'Calculates the Sharpe Ratio for a given stock ticker.',
        'parameters': {
            'type': 'object',
            'properties': {
                'ticker': {'type': 'string', 'description': 'The stock ticker symbol (e.g., AAPL for Apple)'},
                'risk_free_rate': {'type': 'number', 'description': 'The risk-free rate (default is 0.01)'}
            },
            'required': ['ticker']
        }
    },
    {
        'name': 'calculate_sortino_ratio',
        'description': 'Calculates the Sortino Ratio for a given stock ticker.',
        'parameters': {
            'type': 'object',
            'properties': {
                'ticker': {'type': 'string', 'description': 'The stock ticker symbol (e.g., AAPL for Apple)'},
                'risk_free_rate': {'type': 'number', 'description': 'The risk-free rate (default is 0.01)'}
            },
            'required': ['ticker']
        }
    },
    {
        'name': 'calculate_treynor_ratio',
        'description': 'Calculates the Treynor Ratio for a given stock ticker.',
        'parameters': {
            'type': 'object',
            'properties': {
                'ticker': {'type': 'string', 'description': 'The stock ticker symbol (e.g., AAPL for Apple)'},
                'benchmark': {'type': 'string', 'description': 'The benchmark index ticker symbol (default is ^GSPC for S&P 500)'},
                'risk_free_rate': {'type': 'number', 'description': 'The risk-free rate (default is 0.01)'}
            },
            'required': ['ticker']
        }
    },
    {
        'name': 'calculate_calmar_ratio',
        'description': 'Calculates the Calmar Ratio for a given stock ticker.',
        'parameters': {
            'type': 'object',
            'properties': {
                'ticker': {'type': 'string', 'description': 'The stock ticker symbol (e.g., AAPL for Apple)'}
            },
            'required': ['ticker']
        }
    },
    {
        'name': 'calculate_pe_ratio',
        'description': 'Calculates the Price-to-Earnings (P/E) Ratio for a given stock ticker.',
        'parameters': {
            'type': 'object',
            'properties': {
                'ticker': {'type': 'string', 'description': 'The stock ticker symbol (e.g., AAPL for Apple)'}
            },
            'required': ['ticker']
        }
    },
    {
        'name': 'calculate_pb_ratio',
        'description': 'Calculates the Price-to-Book (P/B) Ratio for a given stock ticker.',
        'parameters': {
            'type': 'object',
            'properties': {
                'ticker': {'type': 'string', 'description': 'The stock ticker symbol (e.g., AAPL for Apple)'}
            },
            'required': ['ticker']
        }
    },
    {
        'name': 'calculate_dividend_yield',
        'description': 'Calculates the Dividend Yield for a given stock ticker.',
        'parameters': {
            'type': 'object',
            'properties': {
                'ticker': {'type': 'string', 'description': 'The stock ticker symbol (e.g., AAPL for Apple)'}
            },
            'required': ['ticker']
        }
    },
    {
        'name': 'calculate_eps',
        'description': 'Calculates the Earnings Per Share (EPS) for a given stock ticker.',
        'parameters': {
            'type': 'object',
            'properties': {
                'ticker': {'type': 'string', 'description': 'The stock ticker symbol (e.g., AAPL for Apple)'}
            },
            'required': ['ticker']
        }
    },
    {
        'name': 'calculate_de_ratio',
        'description': 'Calculates the Debt-to-Equity (D/E) Ratio for a given stock ticker.',
        'parameters': {
            'type': 'object',
            'properties': {
                'ticker': {'type': 'string', 'description': 'The stock ticker symbol (e.g., AAPL for Apple)'}
            },
            'required': ['ticker']
        }
    },
    {
        'name': 'calculate_price_to_sales',
        'description': 'Calculates the Price-to-Sales Ratio (P/S) for a given stock ticker.',
        'parameters': {
            'type': 'object',
            'properties': {
                'ticker': {
                    'type': 'string',
                    'description': 'The stock ticker symbol for a company (for example AAPL for Apple)'
                }
            },
            'required': ['ticker'],
        }
    },
    {
        'name': 'calculate_roe',
        'description': 'Calculates the Return on Equity (ROE) for a given stock ticker.',
        'parameters': {
            'type': 'object',
            'properties': {
                'ticker': {
                    'type': 'string',
                    'description': 'The stock ticker symbol for a company (for example AAPL for Apple)'
                }
            },
            'required': ['ticker'],
        }
    },
    {
        'name': 'calculate_roa',
        'description': 'Calculates the Return on Assets (ROA) for a given stock ticker.',
        'parameters': {
            'type': 'object',
            'properties': {
                'ticker': {
                    'type': 'string',
                    'description': 'The stock ticker symbol for a company (for example AAPL for Apple)'
                }
            },
            'required': ['ticker'],
        }
    },
    {
        'name': 'calculate_peg',
        'description': 'Calculates the Price/Earnings to Growth Ratio (PEG) for a given stock ticker.',
        'parameters': {
            'type': 'object',
            'properties': {
                'ticker': {
                    'type': 'string',
                    'description': 'The stock ticker symbol for a company (for example AAPL for Apple)'
                }
            },
            'required': ['ticker'],
        }
    },
    {
        'name': 'calculate_current_ratio',
        'description': 'Calculates the Current Ratio for a given stock ticker.',
        'parameters': {
            'type': 'object',
            'properties': {
                'ticker': {
                    'type': 'string',
                    'description': 'The stock ticker symbol for a company (for example AAPL for Apple)'
                }
            },
            'required': ['ticker'],
        }
    },
    {
        'name': 'calculate_quick_ratio',
        'description': 'Calculates the Quick Ratio (Acid-Test) for a given stock ticker.',
        'parameters': {
            'type': 'object',
            'properties': {
                'ticker': {
                    'type': 'string',
                    'description': 'The stock ticker symbol for a company (for example AAPL for Apple)'
                }
            },
            'required': ['ticker'],
        }
    },
    {
        'name': 'calculate_dividend_payout_ratio',
        'description': 'Calculates the Dividend Payout Ratio for a given stock ticker.',
        'parameters': {
            'type': 'object',
            'properties': {
                'ticker': {
                    'type': 'string',
                    'description': 'The stock ticker symbol for a company (for example AAPL for Apple)'
                }
            },
            'required': ['ticker'],
        }
    },
    {
        'name': 'calculate_free_cash_flow',
        'description': 'Calculates the Free Cash Flow (FCF) for a given stock ticker.',
        'parameters': {
            'type': 'object',
            'properties': {
                'ticker': {
                    'type': 'string',
                    'description': 'The stock ticker symbol for a company (for example AAPL for Apple)'
                }
            },
            'required': ['ticker'],
        }
    },
    {
        'name': 'calculate_gross_margin',
        'description': 'Calculates the Gross Margin for a given stock ticker.',
        'parameters': {
            'type': 'object',
            'properties': {
                'ticker': {
                    'type': 'string',
                    'description': 'The stock ticker symbol for a company (for example AAPL for Apple)'
                }
            },
            'required': ['ticker'],
        }
    },
    {
        'name': 'calculate_operating_margin',
        'description': 'Calculates the Operating Margin for a given stock ticker.',
        'parameters': {
            'type': 'object',
            'properties': {
                'ticker': {
                    'type': 'string',
                    'description': 'The stock ticker symbol for a company (for example AAPL for Apple)'
                }
            },
            'required': ['ticker'],
        }
    }
]

available_functions = {
    'get_stock_price': get_stock_price,
    'calculate_SMA': calculate_SMA,
    'calculate_EMA': calculate_EMA,
    'calculate_RSI': calculate_RSI,
    'calculate_MACD': calculate_MACD,
    'plot_stock_price': plot_stock_price,
    'calculate_bollinger_bands': calculate_bollinger_bands,
    'calculate_stochastic_oscillator': calculate_stochastic_oscillator,
    'calculate_atr': calculate_atr,
    'calculate_obv': calculate_obv,
    'calculate_macd_histogram': calculate_macd_histogram,
    'calculate_vwap': calculate_vwap,
    'calculate_linear_regression_trend': calculate_linear_regression_trend,
    'calculate_adx': calculate_adx,
    'calculate_cci': calculate_cci,
    'calculate_beta': calculate_beta,
    'calculate_sharpe_ratio': calculate_sharpe_ratio,
    'calculate_sortino_ratio': calculate_sortino_ratio,
    'calculate_treynor_ratio': calculate_treynor_ratio,
    'calculate_calmar_ratio': calculate_calmar_ratio,
    'calculate_pe_ratio': calculate_pe_ratio,
    'calculate_pb_ratio': calculate_pb_ratio,
    'calculate_dividend_yield': calculate_dividend_yield,
    'calculate_eps': calculate_eps,
    'calculate_de_ratio': calculate_de_ratio,
    'calculate_price_to_sales': calculate_price_to_sales,
    'calculate_roe': calculate_roe,
    'calculate_roa': calculate_roa,
    'calculate_peg': calculate_peg,
    'calculate_current_ratio': calculate_current_ratio,
    'calculate_quick_ratio': calculate_quick_ratio,
    'calculate_dividend_payout_ratio': calculate_dividend_payout_ratio,
    'calculate_free_cash_flow': calculate_free_cash_flow,
    'calculate_gross_margin': calculate_gross_margin,
    'calculate_operating_margin': calculate_operating_margin
}

def execute_function(function_name, function_args):
    function_map = {f['name']: globals()[f['name']] for f in functions}
    if function_name in function_map:
        return function_map[function_name](**function_args)
    return None

def process_stock_related_query(user_input):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-0613",
            messages=st.session_state['messages'],
            functions=functions,
            function_call='auto'
        )
        response_message = response['choices'][0]['message']
        st.session_state['messages'].append(response_message)
        if response_message.get('function_call'):
            function_name = response_message['function_call']['name']
            function_args = json.loads(response_message['function_call']['arguments'])
            function_response = execute_function(function_name, function_args)
            st.session_state['messages'].append({'role': 'function', 'name': function_name, 'content': function_response})
    except Exception as e:
        st.text('Error occurred: ' + str(e))

def process_user_input(user_input):
    process_stock_related_query(user_input)

    last_message = st.session_state['messages'][-1] if st.session_state['messages'] else None
    if last_message and last_message['role'] == 'function':
        return
    try:
        conversation = retrieval_chain.run(user_input)
        st.text(conversation)
        st.session_state['messages'].append({'role': 'assistant', 'content': conversation})
    except Exception as e:
        st.text('Error occurred: ' + str(e))

    last_message = st.session_state['messages'][-1] if st.session_state['messages'] else None
    if last_message and last_message['role'] == 'assistant':
        return

    try:
        response = openai.Completion.create(
            engine="gpt-4",
            prompt=user_input,
            max_tokens=100
        )
        st.text(response.choices[0].text.strip())
        st.session_state['messages'].append({'role': 'assistant', 'content': response.choices[0].text.strip()})
    except Exception as e:
        st.text('Error occurred: ' + str(e))

st.title('Financial Assistant')
st.text('Ask a question about a stock (e.g., "What is the latest price of AAPL?")')

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

for message in st.session_state['messages']:
    if message['role'] == 'user':
        st.text_area('You:', value=message['content'], key=str(hash(message['content'])))
    else:
        st.text_area('Assistant:', value=message['content'], key=str(hash(message['content'])), height=200)

user_input = st.text_input('Your question:')

if user_input:
    st.session_state['messages'].append({'role': 'user', 'content': user_input})
    process_user_input(user_input)
