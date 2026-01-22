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
def test_for_zero_length_in_choose(self):
    """Ticket #882"""
    a = np.array(1)
    assert_raises(ValueError, lambda x: x.choose([]), a)