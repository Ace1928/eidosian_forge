import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def test_replace_operand(self):
    block = self.block(name='my_block')
    builder = ir.IRBuilder(block)
    a, b = builder.function.args[:2]
    undef1 = ir.Constant(ir.IntType(32), ir.Undefined)
    undef2 = ir.Constant(ir.IntType(32), ir.Undefined)
    c = builder.add(undef1, undef2, 'c')
    self.check_block(block, '            my_block:\n                %"c" = add i32 undef, undef\n            ')
    c.replace_usage(undef1, a)
    c.replace_usage(undef2, b)
    self.check_block(block, '            my_block:\n                %"c" = add i32 %".1", %".2"\n            ')