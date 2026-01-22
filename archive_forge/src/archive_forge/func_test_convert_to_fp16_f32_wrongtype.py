import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def test_convert_to_fp16_f32_wrongtype(self):
    block = self.block(name='my_block')
    builder = ir.IRBuilder(block)
    a = ir.Constant(int16, 5)
    with self.assertRaises(TypeError) as raises:
        builder.convert_to_fp16(a, name='b')
    self.assertIn('expected a float type, got i16', str(raises.exception))