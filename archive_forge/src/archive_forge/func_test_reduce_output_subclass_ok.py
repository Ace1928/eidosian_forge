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
def test_reduce_output_subclass_ok(self):

    class MyArr(np.ndarray):
        pass
    out = np.empty(())
    np.add.reduce(np.ones(5), out=out)
    out = out.view(MyArr)
    assert np.add.reduce(np.ones(5), out=out) is out
    assert type(np.add.reduce(out)) is MyArr