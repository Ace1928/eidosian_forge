import ctypes
import numpy as np
from numba.cuda.cudadrv import driver, drvapi, devices
from numba.cuda.testing import unittest, ContextResettingTestCase
from numba.cuda.testing import skip_on_cudasim
def test_derived_pointer(self):

    def handle_val(mem):
        if driver.USE_NV_BINDING:
            return int(mem.handle)
        else:
            return mem.handle.value

    def check(m, offset):
        v1 = m.view(offset)
        self.assertEqual(handle_val(v1.owner), handle_val(m))
        self.assertEqual(m.refct, 2)
        self.assertEqual(handle_val(v1) - offset, handle_val(v1.owner))
        v2 = v1.view(offset)
        self.assertEqual(handle_val(v2.owner), handle_val(m))
        self.assertEqual(handle_val(v2.owner), handle_val(m))
        self.assertEqual(handle_val(v2) - offset * 2, handle_val(v2.owner))
        self.assertEqual(m.refct, 3)
        del v2
        self.assertEqual(m.refct, 2)
        del v1
        self.assertEqual(m.refct, 1)
    m = self.context.memalloc(1024)
    check(m=m, offset=0)
    check(m=m, offset=1)