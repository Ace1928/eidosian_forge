import sys
import os
import mmap
import pytest
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryFile
from numpy import (
from numpy import arange, allclose, asarray
from numpy.testing import (
def test_no_shape(self):
    self.tmpfp.write(b'a' * 16)
    mm = memmap(self.tmpfp, dtype='float64')
    assert_equal(mm.shape, (2,))