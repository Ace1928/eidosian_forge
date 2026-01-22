import pytest
from pytest import param
from numpy.testing import IS_WASM
import numpy as np
def direct_cast():
    np.array([value, value, value]).astype(dtype)