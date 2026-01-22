from io import StringIO
import re
from string import ascii_uppercase as uppercase
import sys
import textwrap
import numpy as np
import pytest
from pandas.compat import (
from pandas import (
import pandas._testing as tm
def test_info_max_cols():
    df = DataFrame(np.random.randn(10, 5))
    for len_, verbose in [(5, None), (5, False), (12, True)]:
        with option_context('max_info_columns', 4):
            buf = StringIO()
            df.info(buf=buf, verbose=verbose)
            res = buf.getvalue()
            assert len(res.strip().split('\n')) == len_
    for len_, verbose in [(12, None), (5, False), (12, True)]:
        with option_context('max_info_columns', 5):
            buf = StringIO()
            df.info(buf=buf, verbose=verbose)
            res = buf.getvalue()
            assert len(res.strip().split('\n')) == len_
    for len_, max_cols in [(12, 5), (5, 4)]:
        with option_context('max_info_columns', 4):
            buf = StringIO()
            df.info(buf=buf, max_cols=max_cols)
            res = buf.getvalue()
            assert len(res.strip().split('\n')) == len_
        with option_context('max_info_columns', 5):
            buf = StringIO()
            df.info(buf=buf, max_cols=max_cols)
            res = buf.getvalue()
            assert len(res.strip().split('\n')) == len_