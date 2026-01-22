import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def test_convert_from_fp16_f32(self):
    block = self.block(name='my_block')
    builder = ir.IRBuilder(block)
    a = ir.Constant(int16, 5)
    b = builder.convert_from_fp16(a, name='b', to=flt)
    builder.ret(b)
    self.check_block(block, '            my_block:\n                %"b" = call float @"llvm.convert.from.fp16.f32"(i16 5)\n                ret float %"b"\n            ')