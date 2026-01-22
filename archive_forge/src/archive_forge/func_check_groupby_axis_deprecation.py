from __future__ import annotations
import contextlib
import string
import warnings
import numpy as np
import pandas as pd
from packaging.version import Version
import pandas.testing as tm
@contextlib.contextmanager
def check_groupby_axis_deprecation():
    if PANDAS_GE_210:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', ".*Call without passing 'axis' instead|.*Operate on the un-grouped DataFrame instead", FutureWarning)
            yield
    else:
        yield