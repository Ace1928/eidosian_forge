from __future__ import annotations
import contextlib
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
import pytest
from pandas.api.types import is_scalar
import dask.dataframe as dd
from dask.array.numpy_compat import NUMPY_GE_125
from dask.dataframe._compat import (
from dask.dataframe.utils import (
def test_std_raises_on_index():
    with pytest.raises((NotImplementedError, AttributeError), match='`std` is only supported with objects that are Dataframes or Series|has no attribute'):
        dd.from_pandas(pd.DataFrame({'test': [1, 2]}), npartitions=2).index.std()