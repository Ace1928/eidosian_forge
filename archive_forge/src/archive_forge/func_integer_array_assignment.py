import pytest
from pytest import param
from numpy.testing import IS_WASM
import numpy as np
def integer_array_assignment():
    arr = np.empty(3, dtype=dtype)
    values = np.array([value, value])
    arr[[0, 1]] = values