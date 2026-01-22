def tips(pretty_names=False):
    """
    Each row represents a restaurant bill.

    https://vincentarelbundock.github.io/Rdatasets/doc/reshape2/tips.html

    Returns:
        A `pandas.DataFrame` with 244 rows and the following columns:
        `['total_bill', 'tip', 'sex', 'smoker', 'day', 'time', 'size']`."""
    df = _get_dataset('tips')
    if pretty_names:
        df.rename(mapper=dict(total_bill='Total Bill', tip='Tip', sex='Payer Gender', smoker='Smokers at Table', day='Day of Week', time='Meal', size='Party Size'), axis='columns', inplace=True)
    return df