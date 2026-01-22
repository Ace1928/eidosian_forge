import itertools
import math
import sys
from numba import jit, types
from numba.tests.support import TestCase
from .complex_usecases import *
import unittest
def run_unary(self, pyfunc, x_types, x_values, ulps=1, abs_tol=None, flags=enable_pyobj_flags):
    for tx in x_types:
        cfunc = jit((tx,), **flags)(pyfunc)
        prec = 'single' if tx in (types.float32, types.complex64) else 'double'
        for vx in x_values:
            try:
                expected = pyfunc(vx)
            except ValueError as e:
                self.assertIn('math domain error', str(e))
                continue
            got = cfunc(vx)
            msg = 'for input %r with prec %r' % (vx, prec)
            self.assertPreciseEqual(got, expected, prec=prec, ulps=ulps, abs_tol=abs_tol, msg=msg)