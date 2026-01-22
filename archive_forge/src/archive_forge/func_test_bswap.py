import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def test_bswap(self):
    block = self.block(name='my_block')
    builder = ir.IRBuilder(block)
    a = ir.Constant(int32, 5)
    c = builder.bswap(a, name='c')
    builder.ret(c)
    self.check_block(block, '            my_block:\n                %"c" = call i32 @"llvm.bswap.i32"(i32 5)\n                ret i32 %"c"\n            ')