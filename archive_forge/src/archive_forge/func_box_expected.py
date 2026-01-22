from __future__ import annotations
from decimal import Decimal
import operator
import os
from sys import byteorder
from typing import (
import warnings
import numpy as np
from pandas._config.localization import (
from pandas.compat import pa_version_under10p1
from pandas.core.dtypes.common import is_string_dtype
import pandas as pd
from pandas import (
from pandas._testing._io import (
from pandas._testing._warnings import (
from pandas._testing.asserters import (
from pandas._testing.compat import (
from pandas._testing.contexts import (
from pandas.core.arrays import (
from pandas.core.arrays._mixins import NDArrayBackedExtensionArray
from pandas.core.construction import extract_array
def box_expected(expected, box_cls, transpose: bool=True):
    """
    Helper function to wrap the expected output of a test in a given box_class.

    Parameters
    ----------
    expected : np.ndarray, Index, Series
    box_cls : {Index, Series, DataFrame}

    Returns
    -------
    subclass of box_cls
    """
    if box_cls is pd.array:
        if isinstance(expected, RangeIndex):
            expected = NumpyExtensionArray(np.asarray(expected._values))
        else:
            expected = pd.array(expected, copy=False)
    elif box_cls is Index:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'Dtype inference', category=FutureWarning)
            expected = Index(expected)
    elif box_cls is Series:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'Dtype inference', category=FutureWarning)
            expected = Series(expected)
    elif box_cls is DataFrame:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'Dtype inference', category=FutureWarning)
            expected = Series(expected).to_frame()
        if transpose:
            expected = expected.T
            expected = pd.concat([expected] * 2, ignore_index=True)
    elif box_cls is np.ndarray or box_cls is np.array:
        expected = np.array(expected)
    elif box_cls is to_array:
        expected = to_array(expected)
    else:
        raise NotImplementedError(box_cls)
    return expected