import pytest
from pytest import param
from numpy.testing import IS_WASM
import numpy as np
def copyto_scalar_masked():
    arr = np.empty(3, dtype=dtype)
    np.copyto(arr, value, casting='unsafe', where=[True, False, True])