import time
from functools import partial
from io import BytesIO
from itertools import product
from threading import Lock, Thread
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from ..fileslice import (
def test_canonical_slicers():
    slicers = (slice(None), slice(9), slice(0, 9), slice(1, 10), slice(1, 10, 2), 2, np.array(2))
    shape = (10, 10)
    for slice0 in slicers:
        assert canonical_slicers((slice0,), shape) == (slice0, slice(None))
        for slice1 in slicers:
            sliceobj = (slice0, slice1)
            assert canonical_slicers(sliceobj, shape) == sliceobj
            assert canonical_slicers(sliceobj, shape + (2, 3, 4)) == sliceobj + (slice(None),) * 3
            assert canonical_slicers(sliceobj * 3, shape * 3) == sliceobj * 3
            assert canonical_slicers(sliceobj + (None,), shape) == sliceobj + (None,)
            assert canonical_slicers((None,) + sliceobj, shape) == (None,) + sliceobj
            assert canonical_slicers((None,) + sliceobj + (None,), shape) == (None,) + sliceobj + (None,)
    assert canonical_slicers((Ellipsis,), shape) == (slice(None), slice(None))
    assert canonical_slicers((Ellipsis, None), shape) == (slice(None), slice(None), None)
    assert canonical_slicers((Ellipsis, 1), shape) == (slice(None), 1)
    assert canonical_slicers((1, Ellipsis), shape) == (1, slice(None))
    assert canonical_slicers((1, 1, Ellipsis), shape) == (1, 1)
    assert canonical_slicers((1, Ellipsis, 2), (10, 1, 2, 3, 11)) == (1, slice(None), slice(None), slice(None), 2)
    with pytest.raises(ValueError):
        canonical_slicers((Ellipsis, 1, Ellipsis), (2, 3, 4, 5))
    for slice0 in (slice(10), slice(0, 10), slice(0, 10, 1)):
        assert canonical_slicers((slice0, 1), shape) == (slice(None), 1)
    for slice0 in (slice(10), slice(0, 10), slice(0, 10, 1)):
        assert canonical_slicers((slice0, 1), shape) == (slice(None), 1)
        assert canonical_slicers((1, slice0), shape) == (1, slice(None))
    assert canonical_slicers(1, shape) == (1, slice(None))
    assert canonical_slicers(slice(None), shape) == (slice(None), slice(None))
    with pytest.raises(ValueError):
        canonical_slicers((np.array([1]), 1), shape)
    with pytest.raises(ValueError):
        canonical_slicers((1, np.array([1])), shape)
    with pytest.raises(ValueError):
        canonical_slicers((10,), shape)
    with pytest.raises(ValueError):
        canonical_slicers((1, 10), shape)
    with pytest.raises(ValueError):
        canonical_slicers((10,), shape, True)
    with pytest.raises(ValueError):
        canonical_slicers((1, 10), shape, True)
    assert canonical_slicers((10,), shape, False) == (10, slice(None))
    assert canonical_slicers((1, 10), shape, False) == (1, 10)
    assert canonical_slicers(-1, shape) == (9, slice(None))
    assert canonical_slicers((slice(None), -1), shape) == (slice(None), 9)
    assert canonical_slicers(np.array(2), shape) == canonical_slicers(2, shape)
    assert canonical_slicers((np.array(2), np.array(1)), shape) == canonical_slicers((2, 1), shape)
    assert canonical_slicers((2, np.array(1)), shape) == canonical_slicers((2, 1), shape)
    assert canonical_slicers((np.array(2), 1), shape) == canonical_slicers((2, 1), shape)