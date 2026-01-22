import sys
import pytest
import numpy as np
from numpy.testing import assert_array_equal, IS_PYPY
def dlpack_deleter_exception(self):
    x = np.arange(5)
    _ = x.__dlpack__()
    raise RuntimeError