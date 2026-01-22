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
def test_allany_oddities(self):
    store = empty((), dtype=bool)
    full = array([1, 2, 3], mask=True)
    assert_(full.all() is masked)
    full.all(out=store)
    assert_(store)
    assert_(store._mask, True)
    assert_(store is not masked)
    store = empty((), dtype=bool)
    assert_(full.any() is masked)
    full.any(out=store)
    assert_(not store)
    assert_(store._mask, True)
    assert_(store is not masked)