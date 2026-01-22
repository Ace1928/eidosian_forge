import sys
import os
import mmap
import pytest
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryFile
from numpy import (
from numpy import arange, allclose, asarray
from numpy.testing import (
def test_memmap_subclass(self):

    class MemmapSubClass(memmap):
        pass
    fp = MemmapSubClass(self.tmpfp, dtype=self.dtype, shape=self.shape)
    fp[:] = self.data
    assert_(sum(fp, axis=0).__class__ is MemmapSubClass)
    assert_(sum(fp).__class__ is MemmapSubClass)
    assert_(fp[1:, :-1].__class__ is MemmapSubClass)
    assert fp[[0, 1]].__class__ is MemmapSubClass