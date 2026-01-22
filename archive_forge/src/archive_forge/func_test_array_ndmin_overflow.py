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
def test_array_ndmin_overflow(self):
    """Ticket #947."""
    assert_raises(ValueError, lambda: np.array([1], ndmin=33))