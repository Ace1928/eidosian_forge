import warnings
import itertools
import sys
import ctypes as ct
import pytest
from pytest import param
import numpy as np
import numpy.core._umath_tests as umt
import numpy.linalg._umath_linalg as uml
import numpy.core._operand_flag_tests as opflag_tests
import numpy.core._rational_tests as _rational_tests
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
from numpy.compat import pickle
def test_reduce_identity_depends_on_loop(self):
    """
        The type of the result should always depend on the selected loop, not
        necessarily the output (only relevant for object arrays).
        """
    assert type(np.add.reduce([], dtype=object)) is int
    out = np.array(None, dtype=object)
    np.add.reduce([], out=out, dtype=np.float64)
    assert type(out[()]) is float