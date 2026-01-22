import time
from functools import partial
from io import BytesIO
from itertools import product
from threading import Lock, Thread
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from ..fileslice import (
def test_is_fancy():
    slices = (2, [2], [2, 3], Ellipsis, np.array((2, 3)))
    for slice0 in slices:
        _check_slice(slice0)
        _check_slice((slice0,))
        maybe_bad = slice0 is Ellipsis
        for slice1 in slices:
            if maybe_bad and slice1 is Ellipsis:
                continue
            _check_slice((slice0, slice1))
    assert not is_fancy((None,))
    assert not is_fancy((None, 1))
    assert not is_fancy((1, None))
    assert is_fancy(1) is False