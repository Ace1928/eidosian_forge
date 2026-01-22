import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def test_if_then_long_label(self):
    full_label = 'Long' * 20
    block = self.block(name=full_label)
    builder = ir.IRBuilder(block)
    z = ir.Constant(int1, 0)
    a = builder.add(z, z, 'a')
    with builder.if_then(a):
        b = builder.add(z, z, 'b')
        with builder.if_then(b):
            builder.add(z, z, 'c')
    builder.ret_void()
    self.check_func_body(builder.function, '            {full_label}:\n                %"a" = add i1 0, 0\n                br i1 %"a", label %"{label}.if", label %"{label}.endif"\n            {label}.if:\n                %"b" = add i1 0, 0\n                br i1 %"b", label %"{label}.if.if", label %"{label}.if.endif"\n            {label}.endif:\n                ret void\n            {label}.if.if:\n                %"c" = add i1 0, 0\n                br label %"{label}.if.endif"\n            {label}.if.endif:\n                br label %"{label}.endif"\n            '.format(full_label=full_label, label=full_label[:25] + '..'))