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
def test_misaligned_objects_segfault(self):
    a1 = np.zeros((10,), dtype='O,c')
    a2 = np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'], 'S10')
    a1['f0'] = a2
    repr(a1)
    np.argmax(a1['f0'])
    a1['f0'][1] = 'FOO'
    a1['f0'] = 'FOO'
    np.array(a1['f0'], dtype='S')
    np.nonzero(a1['f0'])
    a1.sort()
    copy.deepcopy(a1)