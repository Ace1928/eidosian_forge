import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def test_type_instantiaton(self):
    """
        Instantiating a type should create a constant.
        """
    c = int8(42)
    self.assertIsInstance(c, ir.Constant)
    self.assertEqual(str(c), 'i8 42')
    c = int1(True)
    self.assertIsInstance(c, ir.Constant)
    self.assertEqual(str(c), 'i1 true')
    at = ir.ArrayType(int32, 3)
    c = at([c32(4), c32(5), c32(6)])
    self.assertEqual(str(c), '[3 x i32] [i32 4, i32 5, i32 6]')
    c = at([4, 5, 6])
    self.assertEqual(str(c), '[3 x i32] [i32 4, i32 5, i32 6]')
    c = at(None)
    self.assertEqual(str(c), '[3 x i32] zeroinitializer')
    with self.assertRaises(ValueError):
        at([4, 5, 6, 7])
    st1 = ir.LiteralStructType((flt, int1))
    st2 = ir.LiteralStructType((int32, st1))
    c = st1((1.5, True))
    self.assertEqual(str(c), '{float, i1} {float 0x3ff8000000000000, i1 true}')
    c = st2((42, (1.5, True)))
    self.assertEqual(str(c), '{i32, {float, i1}} {i32 42, {float, i1} {float 0x3ff8000000000000, i1 true}}')