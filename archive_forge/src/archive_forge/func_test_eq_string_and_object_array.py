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
def test_eq_string_and_object_array(self):
    a1 = np.array(['a', 'b'], dtype=object)
    a2 = np.array(['a', 'c'])
    assert_array_equal(a1 == a2, [True, False])
    assert_array_equal(a2 == a1, [True, False])