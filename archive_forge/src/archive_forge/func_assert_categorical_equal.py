from __future__ import annotations
import contextlib
import string
import warnings
import numpy as np
import pandas as pd
from packaging.version import Version
import pandas.testing as tm
def assert_categorical_equal(left, right, *args, **kwargs):
    tm.assert_extension_array_equal(left, right, *args, **kwargs)
    assert isinstance(left.dtype, pd.CategoricalDtype), f'{left} is not categorical dtype'
    assert isinstance(right.dtype, pd.CategoricalDtype), f'{right} is not categorical dtype'