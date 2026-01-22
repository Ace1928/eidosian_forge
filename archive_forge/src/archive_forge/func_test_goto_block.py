import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def test_goto_block(self):
    block = self.block(name='my_block')
    builder = ir.IRBuilder(block)
    a, b = builder.function.args[:2]
    builder.add(a, b, 'c')
    bb_new = builder.append_basic_block(name='foo')
    with builder.goto_block(bb_new):
        builder.fadd(a, b, 'd')
        with builder.goto_entry_block():
            builder.sub(a, b, 'e')
        builder.fsub(a, b, 'f')
        builder.branch(bb_new)
    builder.mul(a, b, 'g')
    with builder.goto_block(bb_new):
        builder.fmul(a, b, 'h')
    self.check_block(block, '            my_block:\n                %"c" = add i32 %".1", %".2"\n                %"e" = sub i32 %".1", %".2"\n                %"g" = mul i32 %".1", %".2"\n            ')
    self.check_block(bb_new, '            foo:\n                %"d" = fadd i32 %".1", %".2"\n                %"f" = fsub i32 %".1", %".2"\n                %"h" = fmul i32 %".1", %".2"\n                br label %"foo"\n            ')