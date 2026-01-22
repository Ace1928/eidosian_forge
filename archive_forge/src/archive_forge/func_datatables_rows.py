import json
import re
import warnings
import numpy as np
import pandas as pd
import pandas.io.formats.format as fmt
def datatables_rows(df, count=None, warn_on_unexpected_types=False):
    """Format the values in the table and return the data, row by row, as requested by DataTables"""
    if count is None or len(df.columns) == count:
        empty_columns = []
    else:
        missing_columns = count - len(df.columns)
        assert missing_columns > 0
        empty_columns = [[None] * len(df)] * missing_columns
    try:
        data = list(zip(*empty_columns + [_format_column(x) for _, x in df.items()]))
        has_bigints = any((x.dtype.kind == 'i' and ((x > JS_MAX_SAFE_INTEGER).any() or (x < JS_MIN_SAFE_INTEGER).any()) for _, x in df.items()))
        js = json.dumps(data, cls=generate_encoder(warn_on_unexpected_types))
    except AttributeError:
        data = list(df.iter_rows())
        import polars as pl
        has_bigints = any((x.dtype in [pl.Int64, pl.UInt64] and ((x > JS_MAX_SAFE_INTEGER).any() or (x < JS_MIN_SAFE_INTEGER).any()) for x in (df[col] for col in df.columns)))
        js = json.dumps(data, cls=generate_encoder(False))
    if has_bigints:
        js = n_suffix_for_bigints(js)
    return js