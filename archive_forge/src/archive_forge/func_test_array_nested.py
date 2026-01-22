import dill
from enum import EnumMeta
import sys
from collections import namedtuple
def test_array_nested():
    try:
        import numpy as np
        x = np.array([1])
        y = (x,)
        assert y == dill.copy(y)
    except ImportError:
        pass