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
@pytest.mark.parametrize('casting', ['unsafe', 'same_kind', 'safe'])
def test_partial_signature_mismatch(self, casting):
    res = np.ldexp(np.float32(1.0), np.int_(2), dtype='d')
    assert res.dtype == 'd'
    res = np.ldexp(np.float32(1.0), np.int_(2), signature=(None, None, 'd'))
    assert res.dtype == 'd'
    with pytest.raises(TypeError):
        np.ldexp(1.0, np.uint64(3), dtype='d')
    with pytest.raises(TypeError):
        np.ldexp(1.0, np.uint64(3), signature=(None, None, 'd'))