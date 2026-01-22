import contextlib
import gc
from itertools import product, cycle
import sys
import warnings
from numbers import Number, Integral
import platform
import numpy as np
from numba import jit, njit, typeof
from numba.core import errors
from numba.tests.support import (TestCase, tag, needs_lapack, needs_blas,
from .matmul_usecase import matmul_usecase
import unittest
def assert_contig_sanity(self, got, expected_contig):
    """
        This checks that in a computed result from numba (array, possibly tuple
        of arrays) all the arrays are contiguous in memory and that they are
        all at least one of "C_CONTIGUOUS" or "F_CONTIGUOUS". The computed
        result of the contiguousness is then compared against a hardcoded
        expected result.

        got: is the computed results from numba
        expected_contig: is "C" or "F" and is the expected type of
                        contiguousness across all input values
                        (and therefore tests).
        """
    if isinstance(got, tuple):
        for a in got:
            self.assert_contig_sanity(a, expected_contig)
    elif not isinstance(got, Number):
        c_contig = got.flags.c_contiguous
        f_contig = got.flags.f_contiguous
        msg = 'Results are not at least one of all C or F contiguous.'
        self.assertTrue(c_contig | f_contig, msg)
        msg = 'Computed contiguousness does not match expected.'
        if expected_contig == 'C':
            self.assertTrue(c_contig, msg)
        elif expected_contig == 'F':
            self.assertTrue(f_contig, msg)
        else:
            raise ValueError('Unknown contig')