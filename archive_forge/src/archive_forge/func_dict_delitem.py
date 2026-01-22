import ctypes
import random
from numba.tests.support import TestCase
from numba import _helperlib, jit, typed, types
from numba.core.config import IS_32BITS
from numba.core.datamodel.models import UniTupleModel
from numba.extending import register_model, typeof_impl, unbox, overload
def dict_delitem(self, key_bytes):
    ix, oldval = self.dict_lookup(key_bytes)
    if ix == DKIX_EMPTY:
        return False
    hashval = hash(key_bytes)
    status = self.tc.numba_dict_delitem(self.dp, hashval, ix)
    self.tc.assertEqual(status, 0)
    return True