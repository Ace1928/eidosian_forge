from __future__ import annotations
import contextlib
import string
import warnings
import numpy as np
import pandas as pd
from packaging.version import Version
import pandas.testing as tm
@contextlib.contextmanager
def check_reductions_runtime_warning():
    if PANDAS_GE_200 and (not PANDAS_GE_201):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='invalid value encountered in double_scalars|Degrees of freedom <= 0 for slice', category=RuntimeWarning)
            yield
    else:
        yield