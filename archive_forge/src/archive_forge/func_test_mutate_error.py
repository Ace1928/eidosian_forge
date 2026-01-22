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
def test_mutate_error(self):
    a = np.dtype('i,i')
    with pytest.raises(ValueError, match='must replace all names at once'):
        a.names = ['f0']
    with pytest.raises(ValueError, match='.*and not string'):
        a.names = ['f0', b'not a unicode name']