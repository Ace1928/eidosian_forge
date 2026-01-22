import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def test_call_metadata(self):
    """
        Function calls with metadata arguments.
        """
    block = self.block(name='my_block')
    builder = ir.IRBuilder(block)
    dbg_declare_ty = ir.FunctionType(ir.VoidType(), [ir.MetaDataType()] * 3)
    dbg_declare = ir.Function(builder.module, dbg_declare_ty, 'llvm.dbg.declare')
    a = builder.alloca(int32, name='a')
    b = builder.module.add_metadata(())
    builder.call(dbg_declare, (a, b, b))
    self.check_block(block, '            my_block:\n                %"a" = alloca i32\n                call void @"llvm.dbg.declare"(metadata i32* %"a", metadata !0, metadata !0)\n            ')