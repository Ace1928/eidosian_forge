def stocks(indexed=False, datetimes=False):
    """
    Each row in this wide dataset represents closing prices from 6 tech stocks in 2018/2019.

    Returns:
        A `pandas.DataFrame` with 100 rows and the following columns:
        `['date', 'GOOG', 'AAPL', 'AMZN', 'FB', 'NFLX', 'MSFT']`.
        If `indexed` is True, the 'date' column is used as the index and the column index
        If `datetimes` is True, the 'date' column will be a datetime column
        is named 'company'"""
    df = _get_dataset('stocks')
    if datetimes:
        df['date'] = df['date'].astype('datetime64[ns]')
    if indexed:
        df = df.set_index('date')
        df.columns.name = 'company'
    return df