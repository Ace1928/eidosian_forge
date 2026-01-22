from numpy.testing import assert_raises
import numpy as np
from .. import all
from .._creation_functions import (
from .._dtypes import float32, float64
from .._array_object import Array
def test_ones_errors():
    ones((1,), device='cpu')
    assert_raises(ValueError, lambda: ones((1,), device='gpu'))
    assert_raises(ValueError, lambda: ones((1,), dtype=int))
    assert_raises(ValueError, lambda: ones((1,), dtype='i'))