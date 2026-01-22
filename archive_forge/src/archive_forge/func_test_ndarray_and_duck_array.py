import inspect
import sys
import os
import tempfile
from io import StringIO
from unittest import mock
import numpy as np
from numpy.testing import (
from numpy.core.overrides import (
from numpy.compat import pickle
import pytest
def test_ndarray_and_duck_array(self):

    class Other:
        __array_function__ = _return_not_implemented
    array = np.array(1)
    other = Other()
    args = _get_implementing_args([other, array])
    assert_equal(list(args), [other, array])
    args = _get_implementing_args([array, other])
    assert_equal(list(args), [array, other])