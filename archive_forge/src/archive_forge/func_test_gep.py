import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def test_gep(self):
    m = self.module()
    tp = ir.LiteralStructType((flt, int1))
    gv = ir.GlobalVariable(m, tp, 'myconstant')
    c = gv.gep([ir.Constant(int32, x) for x in (0, 1)])
    self.assertEqual(str(c), 'getelementptr ({float, i1}, {float, i1}* @"myconstant", i32 0, i32 1)')
    self.assertEqual(c.type, ir.PointerType(int1))
    const = ir.Constant(tp, None)
    with self.assertRaises(TypeError):
        const.gep([ir.Constant(int32, 0)])
    const_ptr = ir.Constant(tp.as_pointer(), None)
    c2 = const_ptr.gep([ir.Constant(int32, 0)])
    self.assertEqual(str(c2), 'getelementptr ({float, i1}, {float, i1}* null, i32 0)')
    self.assertEqual(c.type, ir.PointerType(int1))