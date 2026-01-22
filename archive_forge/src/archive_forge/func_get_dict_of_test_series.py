import math
import string
from datetime import datetime, timedelta
from functools import lru_cache
from itertools import cycle
import numpy as np
import pandas as pd
from .utils import find_package_file
def get_dict_of_test_series(polars=False):
    series = {}
    for df_name, df in get_dict_of_test_dfs().items():
        if len(df.columns) > 6:
            continue
        for col in df.columns:
            if not isinstance(df[col], pd.Series):
                continue
            series['{}.{}'.format(df_name, col)] = df[col]
    if polars:
        import polars as pl
        import pyarrow as pa
        polars_series = {}
        for key in series:
            try:
                polars_series[key] = pl.from_pandas(series[key])
            except (pa.ArrowInvalid, ValueError):
                pass
        polars_series['u32'] = pl.DataFrame({'foo': [1, 1, 3, 1]}).groupby('foo').count()
        return polars_series
    return series