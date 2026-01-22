from numpy.testing import assert_raises
import numpy as np
from .. import all
from .._creation_functions import (
from .._dtypes import float32, float64
from .._array_object import Array
def test_full_errors():
    full((1,), 0, device='cpu')
    assert_raises(ValueError, lambda: full((1,), 0, device='gpu'))
    assert_raises(ValueError, lambda: full((1,), 0, dtype=int))
    assert_raises(ValueError, lambda: full((1,), 0, dtype='i'))