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
def test_series_hash():
    series1 = pd.Series([0.0, 1.0, 2.0, 3.0, 4.0])
    series2 = series1.copy()
    assert hashes_equal(series1, series2)
    series2.iloc[0] = 3.14
    assert not hashes_equal(series1, series2)