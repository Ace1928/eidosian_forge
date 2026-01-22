from numpy.testing import assert_raises
import numpy as np
from .. import all
from .._creation_functions import (
from .._dtypes import float32, float64
from .._array_object import Array
def test_meshgrid_dtype_errors():
    meshgrid()
    meshgrid(asarray([1.0], dtype=float32))
    meshgrid(asarray([1.0], dtype=float32), asarray([1.0], dtype=float32))
    assert_raises(ValueError, lambda: meshgrid(asarray([1.0], dtype=float32), asarray([1.0], dtype=float64)))