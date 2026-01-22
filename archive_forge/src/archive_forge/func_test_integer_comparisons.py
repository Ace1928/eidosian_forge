import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def test_integer_comparisons(self):
    block = self.block(name='my_block')
    builder = ir.IRBuilder(block)
    a, b = builder.function.args[:2]
    builder.icmp_unsigned('==', a, b, 'c')
    builder.icmp_unsigned('!=', a, b, 'd')
    builder.icmp_unsigned('<', a, b, 'e')
    builder.icmp_unsigned('<=', a, b, 'f')
    builder.icmp_unsigned('>', a, b, 'g')
    builder.icmp_unsigned('>=', a, b, 'h')
    builder.icmp_signed('==', a, b, 'i')
    builder.icmp_signed('!=', a, b, 'j')
    builder.icmp_signed('<', a, b, 'k')
    builder.icmp_signed('<=', a, b, 'l')
    builder.icmp_signed('>', a, b, 'm')
    builder.icmp_signed('>=', a, b, 'n')
    with self.assertRaises(ValueError):
        builder.icmp_signed('uno', a, b, 'zz')
    with self.assertRaises(ValueError):
        builder.icmp_signed('foo', a, b, 'zz')
    self.assertFalse(block.is_terminated)
    self.check_block(block, '            my_block:\n                %"c" = icmp eq i32 %".1", %".2"\n                %"d" = icmp ne i32 %".1", %".2"\n                %"e" = icmp ult i32 %".1", %".2"\n                %"f" = icmp ule i32 %".1", %".2"\n                %"g" = icmp ugt i32 %".1", %".2"\n                %"h" = icmp uge i32 %".1", %".2"\n                %"i" = icmp eq i32 %".1", %".2"\n                %"j" = icmp ne i32 %".1", %".2"\n                %"k" = icmp slt i32 %".1", %".2"\n                %"l" = icmp sle i32 %".1", %".2"\n                %"m" = icmp sgt i32 %".1", %".2"\n                %"n" = icmp sge i32 %".1", %".2"\n            ')