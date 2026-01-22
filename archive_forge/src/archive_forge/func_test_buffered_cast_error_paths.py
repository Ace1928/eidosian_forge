import sys
import pytest
import textwrap
import subprocess
import numpy as np
import numpy.core._multiarray_tests as _multiarray_tests
from numpy import array, arange, nditer, all
from numpy.testing import (
def test_buffered_cast_error_paths():
    with pytest.raises(ValueError):
        np.nditer((np.array('a', dtype='S1'),), op_dtypes=['i'], casting='unsafe', flags=['buffered'])
    it = np.nditer((np.array(1, dtype='i'),), op_dtypes=['S1'], op_flags=['writeonly'], casting='unsafe', flags=['buffered'])
    with pytest.raises(ValueError):
        with it:
            buf = next(it)
            buf[...] = 'a'