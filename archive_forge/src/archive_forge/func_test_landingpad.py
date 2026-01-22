import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def test_landingpad(self):
    block = self.block(name='my_block')
    builder = ir.IRBuilder(block)
    lp = builder.landingpad(ir.LiteralStructType([int32, int8.as_pointer()]), 'lp')
    int_typeinfo = ir.GlobalVariable(builder.function.module, int8.as_pointer(), '_ZTIi')
    int_typeinfo.global_constant = True
    lp.add_clause(ir.CatchClause(int_typeinfo))
    lp.add_clause(ir.FilterClause(ir.Constant(ir.ArrayType(int_typeinfo.type, 1), [int_typeinfo])))
    builder.resume(lp)
    self.check_block(block, '            my_block:\n                %"lp" = landingpad {i32, i8*}\n                    catch i8** @"_ZTIi"\n                    filter [1 x i8**] [i8** @"_ZTIi"]\n                resume {i32, i8*} %"lp"\n            ')