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
@suppress_copy_mask_on_assignment
def test_oddfeatures_3(self):
    atest = array([10], mask=True)
    btest = array([20])
    idx = atest.mask
    atest[idx] = btest[idx]
    assert_equal(atest, [20])