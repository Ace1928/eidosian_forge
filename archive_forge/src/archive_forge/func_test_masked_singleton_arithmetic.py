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
def test_masked_singleton_arithmetic(self):
    xm = array(0, mask=1)
    assert_((1 / array(0)).mask)
    assert_((1 + xm).mask)
    assert_((-xm).mask)
    assert_(maximum(xm, xm).mask)
    assert_(minimum(xm, xm).mask)