import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def test_uitofp(self):
    c = ir.Constant(int32, 1).uitofp(flt)
    self.assertEqual(str(c), 'uitofp (i32 1 to float)')