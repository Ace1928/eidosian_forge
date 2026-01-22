import inspect
import sys
import pytest
import numpy as np
from numpy.core import arange
from numpy.testing import assert_, assert_equal, assert_raises_regex
from numpy.lib import deprecate, deprecate_with_doc
import numpy.lib.utils as utils
from io import StringIO
def test_unusual_order_negative_stride(self):
    a = arange(12).reshape(3, 4)
    b = a.T[::-1]
    low, high = utils.byte_bounds(b)
    assert_equal(high - low, b.size * b.itemsize)