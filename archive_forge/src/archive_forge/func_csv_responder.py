from functools import partial
import gzip
from io import BytesIO
import pytest
import pandas.util._test_decorators as td
import pandas as pd
import pandas._testing as tm
def csv_responder(df):
    return df.to_csv(index=False).encode('utf-8')