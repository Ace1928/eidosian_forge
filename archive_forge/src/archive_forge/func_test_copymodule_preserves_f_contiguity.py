import copy
import sys
import gc
import tempfile
import pytest
from os import path
from io import BytesIO
from itertools import chain
import numpy as np
from numpy.testing import (
from numpy.testing._private.utils import _no_tracing, requires_memory
from numpy.compat import asbytes, asunicode, pickle
def test_copymodule_preserves_f_contiguity(self):
    a = np.empty((2, 2), order='F')
    b = copy.copy(a)
    c = copy.deepcopy(a)
    assert_(b.flags.fortran)
    assert_(b.flags.f_contiguous)
    assert_(c.flags.fortran)
    assert_(c.flags.f_contiguous)