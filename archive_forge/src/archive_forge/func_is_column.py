@classmethod
def is_column(cls, data):
    if not _with_pandas():
        return False
    return isinstance(data, _pandas.core.series.Series)