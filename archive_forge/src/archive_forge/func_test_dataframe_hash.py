import datetime as dt
import io
import pathlib
import time
from collections import Counter
import numpy as np
import pandas as pd
import param
import pytest
import requests
from panel.io.cache import _find_hash_func, cache
from panel.io.state import set_curdoc, state
from panel.tests.util import serve_and_wait
def test_dataframe_hash():
    data = {'A': [0.0, 1.0, 2.0, 3.0, 4.0], 'B': [0.0, 1.0, 0.0, 1.0, 0.0], 'C': ['foo1', 'foo2', 'foo3', 'foo4', 'foo5'], 'D': pd.bdate_range('1/1/2009', periods=5)}
    df1, df2 = (pd.DataFrame(data), pd.DataFrame(data))
    assert hashes_equal(df1, df2)
    df2['A'] = df2['A'].values[::-1]
    assert not hashes_equal(df1, df2)