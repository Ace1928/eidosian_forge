import sys
import os
import warnings
import pytest
from io import BytesIO
import numpy as np
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
from numpy.lib import format
def test_write_version():
    f = BytesIO()
    arr = np.arange(1)
    format.write_array(f, arr, version=(1, 0))
    format.write_array(f, arr)
    format.write_array(f, arr, version=None)
    format.write_array(f, arr)
    format.write_array(f, arr, version=(2, 0))
    format.write_array(f, arr)
    bad_versions = [(1, 1), (0, 0), (0, 1), (2, 2), (255, 255)]
    for version in bad_versions:
        with assert_raises_regex(ValueError, 'we only support format version.*'):
            format.write_array(f, arr, version=version)