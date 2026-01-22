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
def test_gh_22556():
    source = np.ma.array([0, [0, 1, 2]], dtype=object)
    deepcopy = copy.deepcopy(source)
    deepcopy[1].append('this should not appear in source')
    assert len(source[1]) == 3