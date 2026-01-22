import collections
import contextlib
import copy
import itertools
import math
import pickle
import sys
from typing import Type
import warnings
from absl.testing import absltest
from absl.testing import parameterized
import ml_dtypes
import numpy as np
def dtype_has_inf(dtype):
    """Determines if the dtype has an `inf` representation."""
    inf = float('inf')
    is_inf = False
    try:
        x = dtype(inf)
        is_inf = np.isinf(x)
    except (OverflowError, ValueError):
        pass
    return is_inf