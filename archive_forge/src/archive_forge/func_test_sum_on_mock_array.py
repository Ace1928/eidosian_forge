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
def test_sum_on_mock_array(self):

    class ArrayProxy:

        def __init__(self, value):
            self.value = value

        def __array_function__(self, *args, **kwargs):
            return self.value.__array_function__(*args, **kwargs)

        def __array__(self, *args, **kwargs):
            return self.value.__array__(*args, **kwargs)
    proxy = ArrayProxy(mock.Mock(spec=ArrayProxy))
    proxy.value.__array_function__.return_value = 1
    result = np.sum(proxy)
    assert_equal(result, 1)
    proxy.value.__array_function__.assert_called_once_with(np.sum, (ArrayProxy,), (proxy,), {})
    proxy.value.__array__.assert_not_called()