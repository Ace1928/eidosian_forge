import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def test_non_nullable_int(self):
    constant = ir.Constant(ir.IntType(32), None).constant
    self.assertEqual(constant, 0)