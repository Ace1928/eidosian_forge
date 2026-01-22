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
def numpy_promote_types(a: Type[np.generic], b: Type[np.generic], float_type: Type[np.generic], next_largest_fp_type: Type[np.generic]) -> Type[np.generic]:
    if a == float_type and b == float_type:
        return float_type
    if a == float_type:
        a = next_largest_fp_type
    if b == float_type:
        b = next_largest_fp_type
    return np.promote_types(a, b)