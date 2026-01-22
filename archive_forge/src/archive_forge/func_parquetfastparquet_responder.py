from functools import partial
import gzip
from io import BytesIO
import pytest
import pandas.util._test_decorators as td
import pandas as pd
import pandas._testing as tm
def parquetfastparquet_responder(df):
    import fsspec
    df.to_parquet('memory://fastparquet_user_agent.parquet', index=False, engine='fastparquet', compression=None)
    with fsspec.open('memory://fastparquet_user_agent.parquet', 'rb') as f:
        return f.read()