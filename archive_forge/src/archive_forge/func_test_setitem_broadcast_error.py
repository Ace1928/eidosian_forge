import decimal
import itertools
import numpy as np
import unittest
from numba import jit, njit, typeof
from numba.core import utils, types, errors
from numba.tests.support import TestCase, tag
from numba.core.typing import arraydecl
from numba.core.types import intp, ellipsis, slice2_type, slice3_type
def test_setitem_broadcast_error(self):
    dst = np.arange(5)
    src = np.arange(10).reshape(2, 5)
    with self.assertRaises(ValueError) as raises:
        setitem_broadcast_usecase(dst, src)
    errmsg = str(raises.exception)
    self.assertEqual('cannot broadcast source array for assignment', errmsg)
    dst = np.arange(5).reshape(1, 5)
    src = np.arange(10).reshape(1, 2, 5)
    with self.assertRaises(ValueError) as raises:
        setitem_broadcast_usecase(dst, src)
    errmsg = str(raises.exception)
    self.assertEqual('cannot assign slice from input of different size', errmsg)
    dst = np.arange(10).reshape(2, 5)
    src = np.arange(4)
    with self.assertRaises(ValueError) as raises:
        setitem_broadcast_usecase(dst, src)
    errmsg = str(raises.exception)
    self.assertEqual('cannot assign slice from input of different size', errmsg)