import typing as py_typing
import unittest
from numba.core import types
from numba.core.errors import TypingError
from numba.core.typing.typeof import typeof
from numba.core.typing.asnumbatype import as_numba_type, AsNumbaTypeRegistry
from numba.experimental.jitclass import jitclass
from numba.tests.support import TestCase
def test_simple_types(self):
    self.assertEqual(as_numba_type(int), self.int_nb_type)
    self.assertEqual(as_numba_type(float), self.float_nb_type)
    self.assertEqual(as_numba_type(complex), self.complex_nb_type)
    self.assertEqual(as_numba_type(str), self.str_nb_type)
    self.assertEqual(as_numba_type(bool), self.bool_nb_type)
    self.assertEqual(as_numba_type(type(None)), self.none_nb_type)