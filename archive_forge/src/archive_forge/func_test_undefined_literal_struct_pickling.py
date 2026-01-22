import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def test_undefined_literal_struct_pickling(self):
    i8 = ir.IntType(8)
    st = ir.Constant(ir.LiteralStructType([i8, i8]), ir.Undefined)
    self.assert_pickle_correctly(st)