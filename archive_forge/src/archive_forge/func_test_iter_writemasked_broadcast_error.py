import sys
import pytest
import textwrap
import subprocess
import numpy as np
import numpy.core._multiarray_tests as _multiarray_tests
from numpy import array, arange, nditer, all
from numpy.testing import (
@pytest.mark.parametrize(['mask', 'mask_axes'], [(None, [-1, 0]), (np.zeros((1, 4), dtype='bool'), [0, 1]), (np.zeros((1, 4), dtype='bool'), None), (np.zeros(4, dtype='bool'), [-1, 0]), (np.zeros((), dtype='bool'), [-1, -1]), (np.zeros((), dtype='bool'), None)])
def test_iter_writemasked_broadcast_error(mask, mask_axes):
    arr = np.zeros((3, 4))
    itflags = ['reduce_ok']
    mask_flags = ['arraymask', 'readwrite', 'allocate']
    a_flags = ['writeonly', 'writemasked']
    if mask_axes is None:
        op_axes = None
    else:
        op_axes = [mask_axes, [0, 1]]
    with assert_raises(ValueError):
        np.nditer((mask, arr), flags=itflags, op_flags=[mask_flags, a_flags], op_axes=op_axes)