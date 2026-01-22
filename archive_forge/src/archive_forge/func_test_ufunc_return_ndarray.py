import sys
import os
import mmap
import pytest
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryFile
from numpy import (
from numpy import arange, allclose, asarray
from numpy.testing import (
def test_ufunc_return_ndarray(self):
    fp = memmap(self.tmpfp, dtype=self.dtype, shape=self.shape)
    fp[:] = self.data
    with suppress_warnings() as sup:
        sup.filter(FutureWarning, 'np.average currently does not preserve')
        for unary_op in [sum, average, prod]:
            result = unary_op(fp)
            assert_(isscalar(result))
            assert_(result.__class__ is self.data[0, 0].__class__)
            assert_(unary_op(fp, axis=0).__class__ is ndarray)
            assert_(unary_op(fp, axis=1).__class__ is ndarray)
    for binary_op in [add, subtract, multiply]:
        assert_(binary_op(fp, self.data).__class__ is ndarray)
        assert_(binary_op(self.data, fp).__class__ is ndarray)
        assert_(binary_op(fp, fp).__class__ is ndarray)
    fp += 1
    assert fp.__class__ is memmap
    add(fp, 1, out=fp)
    assert fp.__class__ is memmap