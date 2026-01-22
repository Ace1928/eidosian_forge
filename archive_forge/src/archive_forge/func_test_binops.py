import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def test_binops(self):
    block = self.block(name='my_block')
    builder = ir.IRBuilder(block)
    a, b, ff = builder.function.args[:3]
    builder.add(a, b, 'c')
    builder.fadd(a, b, 'd')
    builder.sub(a, b, 'e')
    builder.fsub(a, b, 'f')
    builder.mul(a, b, 'g')
    builder.fmul(a, b, 'h')
    builder.udiv(a, b, 'i')
    builder.sdiv(a, b, 'j')
    builder.fdiv(a, b, 'k')
    builder.urem(a, b, 'l')
    builder.srem(a, b, 'm')
    builder.frem(a, b, 'n')
    builder.or_(a, b, 'o')
    builder.and_(a, b, 'p')
    builder.xor(a, b, 'q')
    builder.shl(a, b, 'r')
    builder.ashr(a, b, 's')
    builder.lshr(a, b, 't')
    with self.assertRaises(ValueError) as cm:
        builder.add(a, ff)
    self.assertEqual(str(cm.exception), 'Operands must be the same type, got (i32, double)')
    self.assertFalse(block.is_terminated)
    self.check_block(block, '            my_block:\n                %"c" = add i32 %".1", %".2"\n                %"d" = fadd i32 %".1", %".2"\n                %"e" = sub i32 %".1", %".2"\n                %"f" = fsub i32 %".1", %".2"\n                %"g" = mul i32 %".1", %".2"\n                %"h" = fmul i32 %".1", %".2"\n                %"i" = udiv i32 %".1", %".2"\n                %"j" = sdiv i32 %".1", %".2"\n                %"k" = fdiv i32 %".1", %".2"\n                %"l" = urem i32 %".1", %".2"\n                %"m" = srem i32 %".1", %".2"\n                %"n" = frem i32 %".1", %".2"\n                %"o" = or i32 %".1", %".2"\n                %"p" = and i32 %".1", %".2"\n                %"q" = xor i32 %".1", %".2"\n                %"r" = shl i32 %".1", %".2"\n                %"s" = ashr i32 %".1", %".2"\n                %"t" = lshr i32 %".1", %".2"\n            ')