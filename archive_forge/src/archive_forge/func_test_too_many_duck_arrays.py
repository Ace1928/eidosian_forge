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
def test_too_many_duck_arrays(self):
    namespace = dict(__array_function__=_return_not_implemented)
    types = [type('A' + str(i), (object,), namespace) for i in range(33)]
    relevant_args = [t() for t in types]
    actual = _get_implementing_args(relevant_args[:32])
    assert_equal(actual, relevant_args[:32])
    with assert_raises_regex(TypeError, 'distinct argument types'):
        _get_implementing_args(relevant_args)