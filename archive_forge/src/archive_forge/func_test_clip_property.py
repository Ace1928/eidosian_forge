import sys
import warnings
import itertools
import platform
import pytest
import math
from decimal import Decimal
import numpy as np
from numpy.core import umath
from numpy.random import rand, randint, randn
from numpy.testing import (
from numpy.core._rational_tests import rational
from hypothesis import given, strategies as st
from hypothesis.extra import numpy as hynp
@given(data=st.data(), arr=hynp.arrays(dtype=hynp.integer_dtypes() | hynp.floating_dtypes(), shape=hynp.array_shapes()))
def test_clip_property(self, data, arr):
    """A property-based test using Hypothesis.

        This aims for maximum generality: it could in principle generate *any*
        valid inputs to np.clip, and in practice generates much more varied
        inputs than human testers come up with.

        Because many of the inputs have tricky dependencies - compatible dtypes
        and mutually-broadcastable shapes - we use `st.data()` strategy draw
        values *inside* the test function, from strategies we construct based
        on previous values.  An alternative would be to define a custom strategy
        with `@st.composite`, but until we have duplicated code inline is fine.

        That accounts for most of the function; the actual test is just three
        lines to calculate and compare actual vs expected results!
        """
    numeric_dtypes = hynp.integer_dtypes() | hynp.floating_dtypes()
    in_shapes, result_shape = data.draw(hynp.mutually_broadcastable_shapes(num_shapes=2, base_shape=arr.shape))
    s = numeric_dtypes.flatmap(lambda x: hynp.from_dtype(x, allow_nan=False))
    amin = data.draw(s | hynp.arrays(dtype=numeric_dtypes, shape=in_shapes[0], elements={'allow_nan': False}))
    amax = data.draw(s | hynp.arrays(dtype=numeric_dtypes, shape=in_shapes[1], elements={'allow_nan': False}))
    result = np.clip(arr, amin, amax)
    t = np.result_type(arr, amin, amax)
    expected = np.minimum(amax, np.maximum(arr, amin, dtype=t), dtype=t)
    assert result.dtype == t
    assert_array_equal(result, expected)