import operator
from numpy.testing import assert_raises, suppress_warnings
import numpy as np
import pytest
from .. import ones, asarray, reshape, result_type, all, equal
from .._array_object import Array
from .._dtypes import (
@pytest.mark.parametrize('shape', [(), (5,), (3, 3, 3)])
@pytest.mark.parametrize('index', ['string', False, True])
def test_error_on_invalid_index(shape, index):
    a = ones(shape)
    with pytest.raises(IndexError):
        a[index]