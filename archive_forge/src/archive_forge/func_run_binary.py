import itertools
import math
import sys
from numba import jit, types
from numba.tests.support import TestCase
from .complex_usecases import *
import unittest
def run_binary(self, pyfunc, value_types, values, ulps=1, flags=enable_pyobj_flags):
    for tx, ty in value_types:
        cfunc = jit((tx, ty), **flags)(pyfunc)
        prec = 'single' if set([tx, ty]) & set([types.float32, types.complex64]) else 'double'
        for vx, vy in values:
            try:
                expected = pyfunc(vx, vy)
            except ValueError as e:
                self.assertIn('math domain error', str(e))
                continue
            except ZeroDivisionError:
                continue
            got = cfunc(vx, vy)
            msg = 'for input %r with prec %r' % ((vx, vy), prec)
            self.assertPreciseEqual(got, expected, prec=prec, ulps=ulps, msg=msg)