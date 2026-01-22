import ctypes
import random
from numba.tests.support import TestCase
from numba import _helperlib, jit, typed, types
from numba.core.config import IS_32BITS
from numba.core.datamodel.models import UniTupleModel
from numba.extending import register_model, typeof_impl, unbox, overload
def dict_lookup(self, key_bytes):
    hashval = hash(key_bytes)
    oldval_bytes = ctypes.create_string_buffer(self.valsize)
    ix = self.tc.numba_dict_lookup(self.dp, key_bytes, hashval, oldval_bytes)
    self.tc.assertGreaterEqual(ix, DKIX_EMPTY)
    return (ix, oldval_bytes.value)