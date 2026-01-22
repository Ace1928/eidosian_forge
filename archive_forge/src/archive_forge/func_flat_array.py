from __future__ import annotations
import re
from functools import total_ordering
from packaging.version import Version
import numpy as np
import pandas as pd
from numba import jit
from pandas.api.extensions import (
from numbers import Integral
from pandas.api.types import pandas_dtype, is_extension_array_dtype
@property
def flat_array(self):
    """
        numpy array containing concatenation of all nested arrays

        Returns
        -------
        np.ndarray
        """
    return self._flat_array