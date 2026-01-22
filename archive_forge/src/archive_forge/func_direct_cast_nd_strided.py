import pytest
from pytest import param
from numpy.testing import IS_WASM
import numpy as np
def direct_cast_nd_strided():
    arr = np.full((5, 5, 5), fill_value=value)[:, ::2, :]
    arr.astype(dtype)