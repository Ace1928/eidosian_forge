import warnings
import itertools
import pytest
import numpy as np
from numpy.core.numeric import normalize_axis_tuple
from numpy.testing import (
from numpy.ma.testutils import (
from numpy.ma.core import (
from numpy.ma.extras import (
def test_axis_argument_errors(self):
    msg = 'mask = %s, ndim = %s, axis = %s, overwrite_input = %s'
    for ndmin in range(5):
        for mask in [False, True]:
            x = array(1, ndmin=ndmin, mask=mask)
            args = itertools.product(range(-ndmin, ndmin), [False, True])
            for axis, over in args:
                try:
                    np.ma.median(x, axis=axis, overwrite_input=over)
                except Exception:
                    raise AssertionError(msg % (mask, ndmin, axis, over))
            args = itertools.product([-(ndmin + 1), ndmin], [False, True])
            for axis, over in args:
                try:
                    np.ma.median(x, axis=axis, overwrite_input=over)
                except np.AxisError:
                    pass
                else:
                    raise AssertionError(msg % (mask, ndmin, axis, over))