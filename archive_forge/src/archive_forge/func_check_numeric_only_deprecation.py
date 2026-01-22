from __future__ import annotations
import contextlib
import string
import warnings
import numpy as np
import pandas as pd
from packaging.version import Version
import pandas.testing as tm
@contextlib.contextmanager
def check_numeric_only_deprecation(name=None, show_nuisance_warning: bool=False):
    supported_funcs = ['sum', 'median', 'prod', 'min', 'max', 'std', 'var', 'quantile']
    if name not in supported_funcs and PANDAS_GE_150 and (not PANDAS_GE_200):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='The default value of numeric_only', category=FutureWarning)
            yield
    elif not show_nuisance_warning and name not in supported_funcs and (not PANDAS_GE_150):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Dropping of nuisance columns in DataFrame', category=FutureWarning)
            yield
    else:
        yield