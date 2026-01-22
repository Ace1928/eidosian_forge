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
@pytest.mark.skipif(IS_PYSTON, reason='Pyston disables recursion checking')
def test_dict_recursion(self):
    d = dict(names=['self'], formats=[None], offsets=[0])
    d['formats'][0] = d
    with pytest.raises(RecursionError):
        np.dtype(d)