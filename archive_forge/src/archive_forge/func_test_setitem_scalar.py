import sys
import warnings
import copy
import operator
import itertools
import textwrap
import pytest
from functools import reduce
import numpy as np
import numpy.ma.core
import numpy.core.fromnumeric as fromnumeric
import numpy.core.umath as umath
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
from numpy import ndarray
from numpy.compat import asbytes
from numpy.ma.testutils import (
from numpy.ma.core import (
from numpy.compat import pickle
def test_setitem_scalar(self):
    mask_0d = np.ma.masked_array(1, mask=True)
    arr = np.ma.arange(3)
    arr[0] = mask_0d
    assert_array_equal(arr.mask, [True, False, False])