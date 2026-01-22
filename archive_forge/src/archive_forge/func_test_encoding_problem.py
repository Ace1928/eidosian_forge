import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def test_encoding_problem(self):
    c = ir.Constant(ir.ArrayType(ir.IntType(8), 256), bytearray(range(256)))
    m = self.module()
    gv = ir.GlobalVariable(m, c.type, 'myconstant')
    gv.global_constant = True
    gv.initializer = c
    parsed = llvm.parse_assembly(str(m))
    reparsed = llvm.parse_assembly(str(parsed))
    self.assertEqual(str(parsed), str(reparsed))