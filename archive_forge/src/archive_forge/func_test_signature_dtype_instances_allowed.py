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
@pytest.mark.parametrize('get_kwarg', [param(lambda x: dict(dtype=x), id='dtype'), param(lambda x: dict(signature=(x, None, None)), id='signature')])
def test_signature_dtype_instances_allowed(self, get_kwarg):
    msg = 'The `dtype` and `signature` arguments to ufuncs'
    with pytest.raises(TypeError, match=msg):
        np.add(3, 5, **get_kwarg(np.dtype('int64').newbyteorder()))
    with pytest.raises(TypeError, match=msg):
        np.add(3, 5, **get_kwarg(np.dtype('m8[ns]')))
    with pytest.raises(TypeError, match=msg):
        np.add(3, 5, **get_kwarg('m8[ns]'))