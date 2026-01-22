import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def test_binop_flags(self):
    block = self.block(name='my_block')
    builder = ir.IRBuilder(block)
    a, b = builder.function.args[:2]
    builder.add(a, b, 'c', flags=('nuw',))
    builder.sub(a, b, 'd', flags=['nuw', 'nsw'])
    self.check_block(block, '            my_block:\n                %"c" = add nuw i32 %".1", %".2"\n                %"d" = sub nuw nsw i32 %".1", %".2"\n            ')