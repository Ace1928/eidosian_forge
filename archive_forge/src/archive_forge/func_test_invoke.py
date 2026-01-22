import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def test_invoke(self):
    block = self.block(name='my_block')
    builder = ir.IRBuilder(block)
    a, b = builder.function.args[:2]
    tp_f = ir.FunctionType(flt, (int32, int32))
    f = ir.Function(builder.function.module, tp_f, 'f')
    bb_normal = builder.function.append_basic_block(name='normal')
    bb_unwind = builder.function.append_basic_block(name='unwind')
    builder.invoke(f, (a, b), bb_normal, bb_unwind, 'res_f')
    self.check_block(block, '            my_block:\n                %"res_f" = invoke float @"f"(i32 %".1", i32 %".2")\n                    to label %"normal" unwind label %"unwind"\n            ')