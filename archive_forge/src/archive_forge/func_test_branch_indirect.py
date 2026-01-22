import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def test_branch_indirect(self):
    block = self.block(name='my_block')
    builder = ir.IRBuilder(block)
    bb_1 = builder.function.append_basic_block(name='b_1')
    bb_2 = builder.function.append_basic_block(name='b_2')
    indirectbr = builder.branch_indirect(ir.BlockAddress(builder.function, bb_1))
    indirectbr.add_destination(bb_1)
    indirectbr.add_destination(bb_2)
    self.assertTrue(block.is_terminated)
    self.check_block(block, '            my_block:\n                indirectbr i8* blockaddress(@"my_func", %"b_1"), [label %"b_1", label %"b_2"]\n            ')