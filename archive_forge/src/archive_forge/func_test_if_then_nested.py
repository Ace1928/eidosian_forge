import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def test_if_then_nested(self):
    block = self.block(name='one')
    builder = ir.IRBuilder(block)
    z = ir.Constant(int1, 0)
    a = builder.add(z, z, 'a')
    with builder.if_then(a):
        b = builder.add(z, z, 'b')
        with builder.if_then(b):
            builder.add(z, z, 'c')
    builder.ret_void()
    self.check_func_body(builder.function, '            one:\n                %"a" = add i1 0, 0\n                br i1 %"a", label %"one.if", label %"one.endif"\n            one.if:\n                %"b" = add i1 0, 0\n                br i1 %"b", label %"one.if.if", label %"one.if.endif"\n            one.endif:\n                ret void\n            one.if.if:\n                %"c" = add i1 0, 0\n                br label %"one.if.endif"\n            one.if.endif:\n                br label %"one.endif"\n            ')