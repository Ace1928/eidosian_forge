# Importing necessary packages
import pandas as pd  # For data manipulation and analysis
import requests  # For making HTTP requests to a specified URL
import pandas_ta as ta  # For technical analysis indicators
import matplotlib.pyplot as plt  # For creating static, animated, and interactive visualizations
from termcolor import colored as cl  # For coloring terminal text
import math  # Provides access to mathematical functions
import logging  # For tracking events that happen when some software runs
import nltk  # For natural language processing tasks
from newspaper import Article  # For extracting and parsing news articles
import ccxt  # Cryptocurrency exchange library for connecting to various exchanges
import nltk
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
from typing import Union, Optional, Tuple, List, Dict, Any, Callable
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
import gspread
from oauth2client.client import OAuth2Credentials
from tkinter import Tk, Label, Entry, Button, StringVar
import json
import os
import yfinance as yf
from itertools import product
from concurrent.futures import ThreadPoolExecutor, wait
import time
import yaml
import traceback
from typing import TypeAlias

# Type aliases for improved readability and maintainability
DataFrame: TypeAlias = pd.DataFrame
Series: TypeAlias = pd.Series
Array: TypeAlias = np.ndarray
Scalar: TypeAlias = Union[int, float]
IntOrFloat: TypeAlias = Union[int, float]
StrOrFloat: TypeAlias = Union[str, float]
StrOrInt: TypeAlias = Union[str, int]
StrOrBool: TypeAlias = Union[str, bool]
Timestamp: TypeAlias = pd.Timestamp
Timedelta: TypeAlias = pd.Timedelta
DatetimeIndex: TypeAlias = pd.DatetimeIndex
IntervalType: TypeAlias = str  # e.g., '1d', '1h', '1m', '1s'
PathType: TypeAlias = str
UrlType: TypeAlias = str
ApiKeyType: TypeAlias = str
JsonType: TypeAlias = Dict[str, Any]
SeriesOrDataFrame: TypeAlias = Union[Series, DataFrame]
DataSource: TypeAlias = str  # e.g., 'google_sheets', 'benzinga', 'yahoo'
ExchangeType: TypeAlias = str  # e.g., 'binance', 'coinbase', 'kraken'
SymbolType: TypeAlias = str  # e.g., 'BTC/USD', 'ETH/BTC', 'AAPL'
ConfigType: TypeAlias = Dict[str, Any]
IndicatorFuncType: TypeAlias = Callable[[DataFrame], SeriesOrDataFrame]
StrategyFuncType: TypeAlias = Callable[
    [DataFrame, IntOrFloat], Tuple[IntOrFloat, IntOrFloat]
]
SentimentType: TypeAlias = float
PricePredictionType: TypeAlias = float
PositionType: TypeAlias = Dict[str, Any]
TradeType: TypeAlias = Dict[str, Any]
LogType: TypeAlias = str
RoiType: TypeAlias = float
OptParamsType: TypeAlias = Dict[str, Any]
MonitoringConfigType: TypeAlias = Dict[str, Any]
AlertMessageType: TypeAlias = str
ExceptionType: TypeAlias = Exception
ThreadType: TypeAlias = ThreadPoolExecutor
FutureType: TypeAlias = wait

# Ensuring only the intended functions and classes are available for import
__all__: List[str] = [
    "get_historical_data",
    "implement_strategy",
    "analyze_sentiment",
    "get_order_book",
]

# Downloading necessary NLTK resources
nltk.download("averaged_perceptron_tagger")
nltk.download("vader_lexicon")
nltk.download("punkt")

# Configuring logging to display information about program execution
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Setting global matplotlib parameters for plot size and style
plt.rcParams["figure.figsize"]: Tuple[int, int] = (20, 10)
plt.style.use("fivethirtyeight")


def log_function_call(func: Callable) -> Callable:
    """
    Decorator to log function calls.
    Purpose:
        This decorator logs the entry and exit of a function, providing insights into the function's usage
        and aiding in debugging and monitoring.
    """

    def wrapper(*args, **kwargs) -> Any:
        logging.info(f"Entering {func.__name__}")
        result: Any = func(*args, **kwargs)
        logging.info(f"Exiting {func.__name__}")
        return result

    return wrapper


@log_function_call
def get_historical_data(
    symbol: SymbolType,
    start_date: str,
    end_date: str,
    interval: IntervalType,
    data_source: DataSource = "google_sheets",
    sheet_url: Optional[UrlType] = None,
    benzinga_api_key: Optional[ApiKeyType] = None,
    yahoo_api_key: Optional[ApiKeyType] = None,
) -> DataFrame:
    """
    Fetches historical stock data from the specified data source (Google Sheets, Benzinga API, or Yahoo Finance API).

    Parameters:
    - symbol (SymbolType): The stock symbol to fetch historical data for.
    - start_date (str): The start date for the historical data in YYYY-MM-DD format.
    - end_date (str): The end date for the historical data in YYYY-MM-DD format.
    - interval (IntervalType): The interval for the historical data (e.g., "1d" for daily).
    - data_source (DataSource): The data source to fetch the data from. Options: "google_sheets", "benzinga", "yahoo". Default is "google_sheets".
    - sheet_url (Optional[UrlType]): The URL of the Google Sheets spreadsheet containing the stock data. Required if data_source is "google_sheets".
    - benzinga_api_key (Optional[ApiKeyType]): The Benzinga API key. Required if data_source is "benzinga".
    - yahoo_api_key (Optional[ApiKeyType]): The Yahoo Finance API key. Required if data_source is "yahoo".

    Returns:
    - DataFrame: A DataFrame containing the historical stock data.

    Raises:
    - ValueError: If the required parameters for the selected data source are not provided.
    - Exception: If there is an error fetching data from the specified data source.

    Example:
    >>> get_historical_data("AAPL", "2022-01-01", "2023-06-08", "1d", data_source="google_sheets", sheet_url="https://docs.google.com/spreadsheets/d/.../edit#gid=0")
    """
    try:
        if data_source == "google_sheets":
            if sheet_url is None:
                raise ValueError(
                    "sheet_url is required when data_source is 'google_sheets'"
                )
            df: DataFrame = pd.read_csv(
                sheet_url.replace("/edit#gid=", "/export?format=csv&gid=")
            )
            df.columns = ["date", "close"]
            df["open"] = df["close"]
            df["high"] = df["close"]
            df["low"] = df["close"]
            df["volume"] = 0
        elif data_source == "benzinga":
            if benzinga_api_key is None:
                raise ValueError(
                    "benzinga_api_key is required when data_source is 'benzinga'"
                )
            url: UrlType = "https://api.benzinga.com/api/v2/bars"
            querystring: Dict[str, StrOrInt] = {
                "token": benzinga_api_key,
                "symbols": symbol,
                "from": start_date,
                "to": end_date,
                "interval": interval,
            }
            response: requests.Response = requests.get(url, params=querystring)
            response.raise_for_status()
            hist_json: JsonType = response.json()
            if not (
                hist_json and isinstance(hist_json, list) and "candles" in hist_json[0]
            ):
                raise ValueError(
                    "Unexpected JSON structure received from the Benzinga API."
                )
            df = pd.DataFrame(hist_json[0]["candles"])
        elif data_source == "yahoo":
            if yahoo_api_key is None:
                raise ValueError(
                    "yahoo_api_key is required when data_source is 'yahoo'"
                )
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            querystring: Dict[str, StrOrInt] = {
                "period1": int(pd.to_datetime(start_date).timestamp()),
                "period2": int(pd.to_datetime(end_date).timestamp()),
                "interval": interval,
            }
            headers: Dict[str, str] = {"x-api-key": yahoo_api_key}
            response = requests.get(url, params=querystring, headers=headers)
            response.raise_for_status()
            data: JsonType = response.json()
            if not (
                data
                and "chart" in data
                and "result" in data["chart"]
                and data["chart"]["result"]
            ):
                raise ValueError(
                    "Unexpected JSON structure received from the Yahoo Finance API."
                )
            df = pd.DataFrame(data["chart"]["result"][0]["indicators"]["quote"][0])
            df.index = pd.to_datetime(data["chart"]["result"][0]["timestamp"], unit="s")
            df.index.name = "date"
            df = df.rename(
                columns={
                    "open": "open",
                    "high": "high",
                    "low": "low",
                    "close": "close",
                    "volume": "volume",
                }
            )
        else:
            raise ValueError(f"Unsupported data source: {data_source}")

        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        return df
    except Exception as e:
        logging.error(f"Error fetching historical data from {data_source}: {e}")
        raise


@log_function_call
def donchian_strategy(aapl: DataFrame, investment: IntOrFloat) -> None:
    """
    Implements a trading strategy based on Donchian Channels and prints the results.

    Parameters:
    - aapl (DataFrame): The DataFrame containing AAPL stock data with Donchian Channels.
    - investment (IntOrFloat): The initial investment amount.

    Returns:
    - None

    Example:
    >>> implement_strategy(aapl_df, 10000)
    """
    in_position: bool = False  # Flag to track if currently holding a position
    equity: IntOrFloat = investment  # Current equity
    no_of_shares: int = 0  # Number of shares bought

    try:
        for i in range(3, len(aapl)):
            # Buying condition: stock's high equals the upper Donchian Channel and not currently in position
            if aapl["high"][i] == aapl["dcu"][i] and not in_position:
                no_of_shares = math.floor(equity / aapl.close[i])
                equity -= no_of_shares * aapl.close[i]
                in_position = True
                logging.info(
                    f"BUY: {no_of_shares} Shares are bought at ${aapl.close[i]} on {str(aapl.index[i])[:10]}"
                )
            # Selling condition: stock's low equals the lower Donchian Channel and currently in position
            elif aapl["low"][i] == aapl["dcl"][i] and in_position:
                equity += no_of_shares * aapl.close[i]
                in_position = False
                logging.info(
                    f"SELL: {no_of_shares} Shares are sold at ${aapl.close[i]} on {str(aapl.index[i])[:10]}"
                )
        # Closing position at the end if still in position
        if in_position:
            equity += no_of_shares * aapl.close[i]
            logging.info(
                f"\nClosing position at {aapl.close[i]} on {str(aapl.index[i])[:10]}"
            )
            in_position = False

        # Calculating earnings and ROI
        earning: IntOrFloat = round(equity - investment, 2)
        roi: RoiType = round(earning / investment * 100, 2)
        logging.info(f"EARNING: ${earning} ; ROI: {roi}%")
    except Exception as e:
        logging.error(f"Error implementing strategy: {e}")
        raise


def fetch_and_analyze_article_sentiment(
    keyword: Optional[str] = None, url: Optional[UrlType] = None
) -> Optional[SentimentType]:
    """
    Searches for news articles using a keyword or directly analyzes the sentiment of a news article from a given URL.
    It utilizes both TextBlob and NLTK's SentimentIntensityAnalyzer for a comprehensive sentiment analysis.

    Parameters:
    - keyword (Optional[str]): Keyword to search for relevant news articles.
    - url (Optional[UrlType]): URL of the news article to analyze.

    Returns:
    - Optional[SentimentType]: The average sentiment polarity of fetched articles or the specified article, None if no articles are found or an error occurs.
    """
    news_articles: List[str] = []
    if keyword:
        # Placeholder for fetching news articles based on the keyword
        # This should be replaced with actual code to fetch news content
        # For demonstration purposes, we simulate fetching articles with predefined content
        news_articles = [
            "This is a sample positive news about " + keyword,
            "This is a sample negative news about " + keyword,
        ]
    elif url:
        try:
            article: Article = Article(url)
            article.download()
            article.parse()
            news_articles = [article.text]
        except Exception as e:
            logging.error(f"Error fetching the article from {url}: {e}")
            return None

    if not news_articles:
        logging.error(f"No news articles found for {keyword}")
        return None

    total_sentiment_nltk: float = 0
    total_sentiment_textblob: float = 0
    articles_analyzed: int = 0

    for article_content in news_articles:
        try:
            # Analyzing sentiment using TextBlob
            analysis_textblob: TextBlob = TextBlob(article_content)
            total_sentiment_textblob += analysis_textblob.sentiment.polarity

            # Analyzing sentiment using NLTK's SentimentIntensityAnalyzer
            sia: SentimentIntensityAnalyzer = SentimentIntensityAnalyzer()
            sentiment_score_nltk: Dict[str, float] = sia.polarity_scores(
                article_content
            )
            total_sentiment_nltk += sentiment_score_nltk["compound"]

            articles_analyzed += 1
        except Exception as e:
            logging.error(
                f"Error analyzing sentiment for an article with keyword {keyword} or URL {url}: {e}"
            )

    if articles_analyzed == 0:
        return None

    # Calculating average sentiment from both analyzers
    average_sentiment_nltk: float = total_sentiment_nltk / articles_analyzed
    average_sentiment_textblob: float = total_sentiment_textblob / articles_analyzed
    combined_average_sentiment: float = (
        average_sentiment_nltk + average_sentiment_textblob
    ) / 2

    logging.info(
        f"Average sentiment for {keyword or url} using TextBlob: {average_sentiment_textblob}, NLTK: {average_sentiment_nltk}, Combined: {combined_average_sentiment}"
    )

    return combined_average_sentiment


@log_function_call
def get_order_book(exchange_id: str, symbol: str) -> Optional[Dict[str, Any]]:
    """
    Fetches the real-time order book data for a given symbol from a specified exchange.

    Parameters:
    - exchange_id (str): Identifier for the exchange (e.g., 'binance').
    - symbol (str): Trading pair symbol (e.g., 'BTC/USD').

    Returns:
    - Optional[Dict[str, Any]]: Order book data containing bids and asks, or None if an error occurs.

    Example:
    >>> get_order_book('binance', 'BTC/USD')
    """
    try:
        exchange_class = getattr(ccxt, exchange_id)()
        exchange_class.load_markets()
        order_book = exchange_class.fetch_order_book(symbol)
        return order_book
    except Exception as e:
        logging.error(f"Failed to fetch order book for {symbol} on {exchange_id}: {e}")
        return None


def preprocess_data_for_lstm(
    data: pd.DataFrame,
    feature_columns: List[str],
    target_column: str,
    time_steps: int = 60,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepares data for LSTM model training and prediction by scaling the features and creating sequences of time steps.

    Parameters:
    - data (pd.DataFrame): The DataFrame containing the data.
    - feature_columns (List[str]): List of column names to be used as features.
    - target_column (str): Name of the target column.
    - time_steps (int): The number of time steps to be used for training.

    Returns:
    - Tuple[np.ndarray, np.ndarray]: Tuple containing feature and target data arrays.
    """
    # Selecting the specified features and target column from the DataFrame
    data = data[feature_columns + [target_column]]
    # Initializing the MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    # Scaling the data
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    # Creating sequences of time steps for the LSTM model
    for i in range(time_steps, len(scaled_data)):
        X.append(scaled_data[i - time_steps : i, :-1])  # Features
        y.append(scaled_data[i, -1])  # Target

    return np.array(X), np.array(y)


def build_lstm_model(
    input_shape: Tuple[int, int], units: int = 50, dropout: float = 0.2
) -> Sequential:
    """
    Constructs an LSTM model with specified input shape, number of units, and dropout rate.

    Parameters:
    - input_shape (Tuple[int, int]): The shape of the input data (time steps, features).
    - units (int): The number of LSTM units in each layer.
    - dropout (float): Dropout rate for regularization to prevent overfitting.

    Returns:
    - Sequential: The compiled LSTM model ready for training.
    """
    model = Sequential(
        [
            LSTM(units=units, return_sequences=True, input_shape=input_shape),
            Dropout(dropout),
            LSTM(units=units, return_sequences=False),
            Dropout(dropout),
            Dense(units=1),
        ]
    )
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def train_and_predict_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    epochs: int = 100,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Trains a Long Short-Term Memory (LSTM) model on the provided training data and generates predictions on the test data.

    This function encapsulates the process of constructing an LSTM model using the training data's shape, applying early stopping
    to prevent overfitting during training, and finally, using the trained model to predict outcomes based on the test data.

    Parameters:
    - X_train (np.ndarray): The feature data used for training the model.
    - y_train (np.ndarray): The target data used for training the model.
    - X_test (np.ndarray): The feature data used for generating predictions.
    - epochs (int, optional): The total number of training cycles. Defaults to 100.
    - batch_size (int, optional): The number of samples per gradient update. Defaults to 32.

    Returns:
    - np.ndarray: An array containing the predicted values for the test data.

    Raises:
    - ValueError: If the input data does not match the expected dimensions or types.

    Example:
    >>> X_train, y_train, X_test = np.random.rand(100, 10), np.random.rand(100), np.random.rand(20, 10)
    >>> predictions = train_and_predict_lstm(X_train, y_train, X_test)
    >>> print(predictions.shape)
    """
    # Validate input types and shapes
    if (
        not isinstance(X_train, np.ndarray)
        or not isinstance(y_train, np.ndarray)
        or not isinstance(X_test, np.ndarray)
    ):
        raise ValueError("Input data must be of type np.ndarray.")
    if X_train.ndim != 2 or X_test.ndim != 2:
        raise ValueError("Input feature data must be 2-dimensional.")
    if y_train.ndim != 1:
        raise ValueError("Input target data must be 1-dimensional.")

    # Constructing the LSTM model based on the shape of the training data
    model = build_lstm_model(input_shape=X_train.shape[1:])

    # Initializing early stopping mechanism to monitor the training loss and stop training to mitigate overfitting
    early_stopping_callback = EarlyStopping(monitor="loss", patience=10, verbose=1)

    # Logging the start of the training process
    logging.info(
        f"Starting LSTM model training with {epochs} epochs and batch size of {batch_size}."
    )

    # Training the LSTM model with the specified parameters and early stopping callback
    model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping_callback],
        verbose=1,
    )

    # Logging the completion of the training process
    logging.info("LSTM model training completed.")

    # Generating predictions using the trained model on the test data
    predictions = model.predict(X_test)

    # Logging the prediction process
    logging.info(
        f"Generated predictions on the test data with shape {predictions.shape}."
    )

    return predictions


# Main function to run the program
def main():
    if not os.path.exists("credentials.json"):
        pass
    else:
        credentials_path = "credentials.json"

    # Default values; these could be modified to take user input
    ticker = "NASDAQ:AAPL"
    start_date = "2003-01-01"
    end_date = "2023-01-01"
    sheet_url = get_historical_data(
        ticker, start_date, end_date, "1W", "google_sheets", credentials_path
    )
    print(f"Google Sheet created at {sheet_url}")
    # Example usage of the implement_strategy function
    implement_strategy(aapl, 100000)

    # Comparing with SPY ETF buy/hold return
    spy = get_historical_data("SPY", "1993-01-01", "1W")
    spy_ret = round(
        ((spy.close.iloc[-1] - spy.close.iloc[0]) / spy.close.iloc[0]) * 100, 2
    )
    logging.info(f"SPY ETF buy/hold return: {spy_ret}%")
    # Example usage of the get_historical_data function
    aapl = get_historical_data("AAPL", "1993-01-01", "1W")
    logging.info("Fetched historical data for AAPL.")
    # CALCULATING DONCHIAN CHANNEL
    # Adding Donchian Channel columns to the DataFrame
    aapl[["dcl", "dcm", "dcu"]] = aapl.ta.donchian(lower_length=40, upper_length=50)
    # Dropping rows with missing values and the 'time' column, renaming 'dateTime' to 'date'
    aapl = aapl.dropna().drop("time", axis=1).rename(columns={"dateTime": "date"})
    # Setting the 'date' column as the index and converting it to datetime format
    aapl = aapl.set_index("date")
    aapl.index = pd.to_datetime(aapl.index)

    logging.info("Calculated Donchian Channels for AAPL.")

    # PLOTTING DONCHIAN CHANNEL
    # Plotting the close price and Donchian Channels
    plt.plot(aapl[-300:].close, label="CLOSE")
    plt.plot(aapl[-300:].dcl, color="black", linestyle="--", alpha=0.3)
    plt.plot(aapl[-300:].dcm, color="orange", label="DCM")
    plt.plot(aapl[-300:].dcu, color="black", linestyle="--", alpha=0.3, label="DCU,DCL")
    plt.legend()
    plt.title("AAPL DONCHIAN CHANNELS 50")
    plt.xlabel("Date")
    plt.ylabel("Close")
    plt.show()

    logging.info("Plotted Donchian Channels for AAPL.")


if __name__ == "__main__":
    main()


"""
We will be exclusively and entirely focusing on the highlighted code to be edited, utilising all of the code above of the trader.py module/model/program to assist in ensuring all improved and output code adheres to the following principles and guidelines:

Perfectly and completely and fully and entirely and fully integrate and align and implement all parts to the highest standards possible in all regards and details and aspects possible ensuring no loss of any current functionality or detail in any way and meticulous documentation type annotation type handling type conversion doc strings constructors in line comments multi line comments efficient advanced complex concrete logic and flexible robust adaptive dynamic intelligent functional fully implemented complete code in all regards. Ensuring every aspect of the program from start to finish processed and output as specified in perfect alignment perfectly integrated all parts utilised perfectly.
Reiterate over the entire code start to finish and ensure it is meticulouslty and perfectly type hinted and type annotated and all types are explicitly handled even if expected to be correct and type aliasing used along with all other useful aliases to ensure that all type, value, key errors are completely avoided.
Everything retained perfectly functionally and all details.
Only improving or adding information or extending or enhancing and ensuring no loss or simplification or omission of any function or utility or detail in any way.
Ensuring every aspect of the code is elevated to the peak of pythonic perfection every way possible.
Take your time.
Be methodical and systematic.
Use your chain of thought reasoning to work it out from the ground up covering everything verbatim.
Ensuring absolutely no functionality or utility are lost or simplified while doing so.
No simplification. No omission. No summarisation. No placeholders. No deletions. No truncations. No brevity.
Ensuring every single aspect output to the highest standards possible in all aspects and regards and details possible. 
Meticulously and systematically and perfectly and completely. 
Fully implemented complete code. 
Output verbatim in its entirety as specified perfectly exclusively for the code selected that we are working on, not outputting anything that is already present in this instruction as it is already implemented in the program. 
Maximum Complexity. Advanced programming. Innovative deep complex concrete functional logic. 
Verbose. Detailed. Functional. Adaptive. Flexible. Robust. Complete. Entire. 
Fully Implemented. 
No simplifications. 
No truncations. 
No omissions. 
No subtractions. 
No deletions. 
Ensuring no loss of any function or detail in any way at all including maintaining all documentation, type annotation, type aliasing, type hinting, type handling, error handling, logging for all aspects verbosely and specifically and exhaustively. 
Exclusively for the selected code and utilising the rest of the code as context/detail/guidance to ensure perfect fully implemented code output ready for functional deployment and testing with every aspect of the code fully and perfectly aligned and output to the highest possible standards in all regards and details perfectly and completely as specified.
"""
