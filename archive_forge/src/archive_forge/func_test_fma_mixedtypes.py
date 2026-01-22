import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def test_fma_mixedtypes(self):
    block = self.block(name='my_block')
    builder = ir.IRBuilder(block)
    a = ir.Constant(flt, 5)
    b = ir.Constant(dbl, 1)
    c = ir.Constant(flt, 2)
    with self.assertRaises(TypeError) as raises:
        builder.fma(a, b, c, name='fma')
    self.assertIn('expected types to be the same, got float, double, float', str(raises.exception))