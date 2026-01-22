from __future__ import annotations
import contextlib
import string
import warnings
import numpy as np
import pandas as pd
from packaging.version import Version
import pandas.testing as tm
@contextlib.contextmanager
def check_nuisance_columns_warning():
    if not PANDAS_GE_150:
        with warnings.catch_warnings(record=True):
            warnings.filterwarnings('ignore', 'Dropping of nuisance columns', FutureWarning)
            yield
    else:
        yield