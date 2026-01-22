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
def compare_matrix_multiply_results(self, tp):
    d1 = np.array(np.random.rand(2, 3, 4), dtype=tp)
    d2 = np.array(np.random.rand(2, 3, 4), dtype=tp)
    msg = 'matrix multiply on type %s' % d1.dtype.name

    def permute_n(n):
        if n == 1:
            return ([0],)
        ret = ()
        base = permute_n(n - 1)
        for perm in base:
            for i in range(n):
                new = perm + [n - 1]
                new[n - 1] = new[i]
                new[i] = n - 1
                ret += (new,)
        return ret

    def slice_n(n):
        if n == 0:
            return ((),)
        ret = ()
        base = slice_n(n - 1)
        for sl in base:
            ret += (sl + (slice(None),),)
            ret += (sl + (slice(0, 1),),)
        return ret

    def broadcastable(s1, s2):
        return s1 == s2 or s1 == 1 or s2 == 1
    permute_3 = permute_n(3)
    slice_3 = slice_n(3) + ((slice(None, None, -1),) * 3,)
    ref = True
    for p1 in permute_3:
        for p2 in permute_3:
            for s1 in slice_3:
                for s2 in slice_3:
                    a1 = d1.transpose(p1)[s1]
                    a2 = d2.transpose(p2)[s2]
                    ref = ref and a1.base is not None
                    ref = ref and a2.base is not None
                    if a1.shape[-1] == a2.shape[-2] and broadcastable(a1.shape[0], a2.shape[0]):
                        assert_array_almost_equal(umt.matrix_multiply(a1, a2), np.sum(a2[..., np.newaxis].swapaxes(-3, -1) * a1[..., np.newaxis, :], axis=-1), err_msg=msg + ' %s %s' % (str(a1.shape), str(a2.shape)))
    assert_equal(ref, True, err_msg='reference check')