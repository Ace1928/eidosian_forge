import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def test_binop_fastmath_flags(self):
    block = self.block(name='my_block')
    builder = ir.IRBuilder(block)
    a, b = builder.function.args[:2]
    builder.fadd(a, b, 'c', flags=('fast',))
    builder.fsub(a, b, 'd', flags=['ninf', 'nsz'])
    self.check_block(block, '            my_block:\n                %"c" = fadd fast i32 %".1", %".2"\n                %"d" = fsub ninf nsz i32 %".1", %".2"\n            ')