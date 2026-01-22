import sys
import pytest
import textwrap
import subprocess
import numpy as np
import numpy.core._multiarray_tests as _multiarray_tests
from numpy import array, arange, nditer, all
from numpy.testing import (
def test_iter_copy():
    a = arange(24).reshape(2, 3, 4)
    i = nditer(a)
    j = i.copy()
    assert_equal([x[()] for x in i], [x[()] for x in j])
    i.iterindex = 3
    j = i.copy()
    assert_equal([x[()] for x in i], [x[()] for x in j])
    i = nditer(a, ['buffered', 'ranged'], order='F', buffersize=3)
    j = i.copy()
    assert_equal([x[()] for x in i], [x[()] for x in j])
    i.iterindex = 3
    j = i.copy()
    assert_equal([x[()] for x in i], [x[()] for x in j])
    i.iterrange = (3, 9)
    j = i.copy()
    assert_equal([x[()] for x in i], [x[()] for x in j])
    i.iterrange = (2, 18)
    next(i)
    next(i)
    j = i.copy()
    assert_equal([x[()] for x in i], [x[()] for x in j])
    with nditer(a, ['buffered'], order='F', casting='unsafe', op_dtypes='f8', buffersize=5) as i:
        j = i.copy()
    assert_equal([x[()] for x in j], a.ravel(order='F'))
    a = arange(24, dtype='<i4').reshape(2, 3, 4)
    with nditer(a, ['buffered'], order='F', casting='unsafe', op_dtypes='>f8', buffersize=5) as i:
        j = i.copy()
    assert_equal([x[()] for x in j], a.ravel(order='F'))