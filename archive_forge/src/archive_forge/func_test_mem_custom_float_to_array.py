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
def test_mem_custom_float_to_array(self):

    class MyFloat:

        def __float__(self):
            return 1.0
    tmp = np.atleast_1d([MyFloat()])
    tmp.astype(float)