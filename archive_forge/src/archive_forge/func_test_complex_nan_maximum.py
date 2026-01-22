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
def test_complex_nan_maximum(self):
    cnan = complex(0, np.nan)
    assert_equal(np.maximum(1, cnan), cnan)