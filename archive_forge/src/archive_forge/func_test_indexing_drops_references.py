import sys
import os
import mmap
import pytest
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryFile
from numpy import (
from numpy import arange, allclose, asarray
from numpy.testing import (
def test_indexing_drops_references(self):
    fp = memmap(self.tmpfp, dtype=self.dtype, mode='w+', shape=self.shape)
    tmp = fp[(1, 2), (2, 3)]
    if isinstance(tmp, memmap):
        assert_(tmp._mmap is not fp._mmap)