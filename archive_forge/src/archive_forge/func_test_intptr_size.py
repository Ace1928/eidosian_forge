from __future__ import absolute_import, division, print_function
import ctypes
import unittest
import pytest
from datashader import datashape
from datashader.datashape import dshape, error, DataShape, Record
@pytest.mark.parametrize('kind', ['intptr', 'uintptr'])
def test_intptr_size(kind):
    assert dshape(kind) == dshape(pointer_sizes[ctypes.sizeof(ctypes.c_void_p)][kind])