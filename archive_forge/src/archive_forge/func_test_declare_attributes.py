import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def test_declare_attributes(self):
    func = self.function()
    func.attributes.add('optsize')
    func.attributes.add('alwaysinline')
    func.attributes.add('convergent')
    func.attributes.alignstack = 16
    tp_pers = ir.FunctionType(int8, (), var_arg=True)
    pers = ir.Function(self.module(), tp_pers, '__gxx_personality_v0')
    func.attributes.personality = pers
    asm = self.descr(func).strip()
    self.assertEqual(asm, 'declare %s alwaysinline convergent optsize alignstack(16) personality i8 (...)* @"__gxx_personality_v0"' % self.proto)
    self.assert_pickle_correctly(func)