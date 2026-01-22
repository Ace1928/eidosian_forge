import ctypes
import random
from numba.tests.support import TestCase
from numba import _helperlib, jit, typed, types
from numba.core.config import IS_32BITS
from numba.core.datamodel.models import UniTupleModel
from numba.extending import register_model, typeof_impl, unbox, overload
def dict_iter_next(self, itptr):
    bk = ctypes.c_void_p(0)
    bv = ctypes.c_void_p(0)
    status = self.tc.numba_dict_iter_next(itptr, ctypes.byref(bk), ctypes.byref(bv))
    if status == -2:
        raise ValueError('dictionary mutated')
    elif status == -3:
        return
    else:
        self.tc.assertGreaterEqual(status, 0)
        self.tc.assertEqual(bk.value % ALIGN, 0, msg='key not aligned')
        self.tc.assertEqual(bv.value % ALIGN, 0, msg='val not aligned')
        key = (ctypes.c_char * self.keysize).from_address(bk.value)
        val = (ctypes.c_char * self.valsize).from_address(bv.value)
        return (key.value, val.value)