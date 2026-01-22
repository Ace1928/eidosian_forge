import unittest
from numba import jit, njit, objmode, typeof, literally
from numba.extending import overload
from numba.core import types
from numba.core.errors import UnsupportedError
from numba.tests.support import (
@skip_unless_py10_or_later
def test_fuse_twice_literal_values(self):
    """
        Tests that the correct literal values are generated
        for a dictionary that produces two DICT_UPDATE
        bytecode entries for the same dictionary.
        """

    def bar(d):
        ...

    @overload(bar)
    def ol_bar(d):
        a = {'a1': 1, 'a2': 2, 'a3': 3, 'a4': 4, 'a5': 5, 'a6': 6, 'a7': 7, 'a8': 8, 'a9': 9, 'a10': 10, 'a11': 11, 'a12': 12, 'a13': 13, 'a14': 14, 'a15': 15, 'a16': 16, 'a17': 17, 'a18': 18, 'a19': 19, 'a20': 20, 'a21': 21, 'a22': 22, 'a23': 23, 'a24': 24, 'a25': 25, 'a26': 26, 'a27': 27, 'a28': 28, 'a29': 29, 'a30': 30, 'a31': 31, 'a32': 32, 'a33': 33, 'a34': 34, 'a35': 35}
        if d.initial_value is None:
            return lambda d: literally(d)
        self.assertTrue(isinstance(d, types.DictType))
        self.assertEqual(d.initial_value, a)
        return lambda d: d

    @njit
    def foo():
        d = {'a1': 1, 'a2': 2, 'a3': 3, 'a4': 4, 'a5': 5, 'a6': 6, 'a7': 7, 'a8': 8, 'a9': 9, 'a10': 10, 'a11': 11, 'a12': 12, 'a13': 13, 'a14': 14, 'a15': 15, 'a16': 16, 'a17': 17, 'a18': 18, 'a19': 19, 'a20': 20, 'a21': 21, 'a22': 22, 'a23': 23, 'a24': 24, 'a25': 25, 'a26': 26, 'a27': 27, 'a28': 28, 'a29': 29, 'a30': 30, 'a31': 31, 'a32': 32, 'a33': 33, 'a34': 34, 'a35': 35}
        bar(d)
    foo()