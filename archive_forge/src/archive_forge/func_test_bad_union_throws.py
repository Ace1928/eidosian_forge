import typing as py_typing
import unittest
from numba.core import types
from numba.core.errors import TypingError
from numba.core.typing.typeof import typeof
from numba.core.typing.asnumbatype import as_numba_type, AsNumbaTypeRegistry
from numba.experimental.jitclass import jitclass
from numba.tests.support import TestCase
def test_bad_union_throws(self):
    bad_unions = [py_typing.Union[str, int], py_typing.Union[int, type(None), py_typing.Tuple[bool, bool]]]
    for bad_py_type in bad_unions:
        with self.assertRaises(TypingError) as raises:
            as_numba_type(bad_py_type)
        self.assertIn('Cannot type Union', str(raises.exception))