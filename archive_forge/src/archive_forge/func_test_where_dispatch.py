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
def test_where_dispatch(self):

    class DuckArray:

        def __array_function__(self, ufunc, method, *inputs, **kwargs):
            return 'overridden'
    array = np.array(1)
    duck_array = DuckArray()
    result = np.std(array, where=duck_array)
    assert_equal(result, 'overridden')