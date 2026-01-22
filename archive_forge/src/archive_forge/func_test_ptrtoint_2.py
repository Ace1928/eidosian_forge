import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def test_ptrtoint_2(self):
    m = self.module()
    gv = ir.GlobalVariable(m, int32, 'myconstant')
    c = gv.ptrtoint(int64)
    self.assertEqual(str(c), 'ptrtoint (i32* @"myconstant" to i64)')
    self.assertRaisesRegex(TypeError, "can only ptrtoint\\(\\) to integer type, not 'i64\\*'", gv.ptrtoint, int64.as_pointer())
    c2 = ir.Constant(int32, 0)
    self.assertRaisesRegex(TypeError, "can only call ptrtoint\\(\\) on pointer type, not 'i32'", c2.ptrtoint, int64)