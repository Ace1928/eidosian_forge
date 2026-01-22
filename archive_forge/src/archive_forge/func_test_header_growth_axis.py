import sys
import os
import warnings
import pytest
from io import BytesIO
import numpy as np
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
from numpy.lib import format
def test_header_growth_axis():
    for is_fortran_array, dtype_space, expected_header_length in [[False, 22, 128], [False, 23, 192], [True, 23, 128], [True, 24, 192]]:
        for size in [10 ** i for i in range(format.GROWTH_AXIS_MAX_DIGITS)]:
            fp = BytesIO()
            format.write_array_header_1_0(fp, {'shape': (2, size) if is_fortran_array else (size, 2), 'fortran_order': is_fortran_array, 'descr': np.dtype([(' ' * dtype_space, int)])})
            assert len(fp.getvalue()) == expected_header_length