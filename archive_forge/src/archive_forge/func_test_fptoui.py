import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def test_fptoui(self):
    c = ir.Constant(flt, 1).fptoui(int32)
    self.assertEqual(str(c), 'fptoui (float 0x3ff0000000000000 to i32)')