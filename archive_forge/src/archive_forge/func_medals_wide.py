def medals_wide(indexed=False):
    """
    This dataset represents the medal table for Olympic Short Track Speed Skating for the
    top three nations as of 2020.

    Returns:
        A `pandas.DataFrame` with 3 rows and the following columns:
        `['nation', 'gold', 'silver', 'bronze']`.
        If `indexed` is True, the 'nation' column is used as the index and the column index
        is named 'medal'"""
    df = _get_dataset('medals')
    if indexed:
        df = df.set_index('nation')
        df.columns.name = 'medal'
    return df