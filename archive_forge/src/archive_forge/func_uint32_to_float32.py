import os
from os.path import join
import sys
import numpy as np
from numpy.testing import (assert_equal, assert_allclose, assert_array_equal,
import pytest
from numpy.random import (
from numpy.random._common import interface
def uint32_to_float32(u):
    return ((u >> np.uint32(8)) * (1.0 / 2 ** 24)).astype(np.float32)