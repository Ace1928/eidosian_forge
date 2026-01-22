import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def test_zext(self):
    c = ir.Constant(int32, 1).zext(int64)
    self.assertEqual(str(c), 'zext (i32 1 to i64)')