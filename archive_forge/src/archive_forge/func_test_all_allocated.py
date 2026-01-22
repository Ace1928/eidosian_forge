import sys
import pytest
import textwrap
import subprocess
import numpy as np
import numpy.core._multiarray_tests as _multiarray_tests
from numpy import array, arange, nditer, all
from numpy.testing import (
def test_all_allocated():
    i = np.nditer([None], op_dtypes=['int64'])
    assert i.operands[0].shape == ()
    assert i.dtypes == (np.dtype('int64'),)
    i = np.nditer([None], op_dtypes=['int64'], itershape=(2, 3, 4))
    assert i.operands[0].shape == (2, 3, 4)