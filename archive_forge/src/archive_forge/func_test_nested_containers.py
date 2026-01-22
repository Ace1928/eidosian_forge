import typing as py_typing
import unittest
from numba.core import types
from numba.core.errors import TypingError
from numba.core.typing.typeof import typeof
from numba.core.typing.asnumbatype import as_numba_type, AsNumbaTypeRegistry
from numba.experimental.jitclass import jitclass
from numba.tests.support import TestCase
def test_nested_containers(self):
    IntList = py_typing.List[int]
    self.assertEqual(as_numba_type(py_typing.List[IntList]), types.ListType(types.ListType(self.int_nb_type)))
    self.assertEqual(as_numba_type(py_typing.List[py_typing.Dict[float, bool]]), types.ListType(types.DictType(self.float_nb_type, self.bool_nb_type)))
    self.assertEqual(as_numba_type(py_typing.Set[py_typing.Tuple[py_typing.Optional[int], float]]), types.Set(types.Tuple([types.Optional(self.int_nb_type), self.float_nb_type])))