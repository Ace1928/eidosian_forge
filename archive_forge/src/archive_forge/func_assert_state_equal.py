import os
from os.path import join
import sys
import numpy as np
from numpy.testing import (assert_equal, assert_allclose, assert_array_equal,
import pytest
from numpy.random import (
from numpy.random._common import interface
def assert_state_equal(actual, target):
    for key in actual:
        if isinstance(actual[key], dict):
            assert_state_equal(actual[key], target[key])
        elif isinstance(actual[key], np.ndarray):
            assert_array_equal(actual[key], target[key])
        else:
            assert actual[key] == target[key]