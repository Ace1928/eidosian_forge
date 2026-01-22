import multiprocessing
import platform
import threading
import pickle
import weakref
from itertools import chain
from io import StringIO
import numpy as np
from numba import njit, jit, typeof, vectorize
from numba.core import types, errors
from numba import _dispatcher
from numba.tests.support import TestCase, captured_stdout
from numba.np.numpy_support import as_dtype
from numba.core.dispatcher import Dispatcher
from numba.extending import overload
from numba.tests.support import needs_lapack, SerialMixin
from numba.testing.main import _TIMEOUT as _RUNNER_TIMEOUT
import unittest
@needs_lapack
@unittest.skipIf(_is_armv7l, 'Unaligned loads unsupported')
def test_misaligned_high_dimension_array_dispatch(self):

    def foo(a):
        return np.linalg.matrix_power(a[0, 0, 0, 0, :, :], 1)
    jitfoo = jit(nopython=True)(foo)

    def check_properties(arr, layout, aligned):
        self.assertEqual(arr.flags.aligned, aligned)
        if layout == 'C':
            self.assertEqual(arr.flags.c_contiguous, True)
        if layout == 'F':
            self.assertEqual(arr.flags.f_contiguous, True)
    n = 729
    r = 3
    dt = np.int8
    count = np.complex128().itemsize // dt().itemsize
    tmp = np.arange(n * count + 1, dtype=dt)
    C_contig_aligned = tmp[:-1].view(np.complex128).reshape(r, r, r, r, r, r)
    check_properties(C_contig_aligned, 'C', True)
    C_contig_misaligned = tmp[1:].view(np.complex128).reshape(r, r, r, r, r, r)
    check_properties(C_contig_misaligned, 'C', False)
    F_contig_aligned = C_contig_aligned.T
    check_properties(F_contig_aligned, 'F', True)
    F_contig_misaligned = C_contig_misaligned.T
    check_properties(F_contig_misaligned, 'F', False)

    def check(name, a):
        a[:, :] = np.arange(n, dtype=np.complex128).reshape(r, r, r, r, r, r)
        expected = foo(a)
        got = jitfoo(a)
        np.testing.assert_allclose(expected, got)
    check('F_contig_misaligned', F_contig_misaligned)
    check('C_contig_aligned', C_contig_aligned)
    check('F_contig_aligned', F_contig_aligned)
    check('C_contig_misaligned', C_contig_misaligned)