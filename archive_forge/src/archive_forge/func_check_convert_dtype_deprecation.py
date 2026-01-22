from __future__ import annotations
import contextlib
import string
import warnings
import numpy as np
import pandas as pd
from packaging.version import Version
import pandas.testing as tm
@contextlib.contextmanager
def check_convert_dtype_deprecation():
    if PANDAS_GE_210:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='the convert_dtype parameter', category=FutureWarning)
            yield
    else:
        yield