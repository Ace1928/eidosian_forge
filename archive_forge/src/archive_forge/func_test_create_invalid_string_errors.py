import sys
import operator
import pytest
import ctypes
import gc
import types
from typing import Any
import numpy as np
import numpy.dtypes
from numpy.core._rational_tests import rational
from numpy.core._multiarray_tests import create_custom_field_dtype
from numpy.testing import (
from numpy.compat import pickle
from itertools import permutations
import random
import hypothesis
from hypothesis.extra import numpy as hynp
def test_create_invalid_string_errors(self):
    one_too_big = np.iinfo(np.intc).max + 1
    with pytest.raises(TypeError):
        type(np.dtype('U'))(one_too_big // 4)
    with pytest.raises(TypeError):
        type(np.dtype('U'))(np.iinfo(np.intp).max // 4 + 1)
    if one_too_big < sys.maxsize:
        with pytest.raises(TypeError):
            type(np.dtype('S'))(one_too_big)
    with pytest.raises(ValueError):
        type(np.dtype('U'))(-1)