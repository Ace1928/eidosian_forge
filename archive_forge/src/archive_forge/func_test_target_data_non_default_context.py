import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def test_target_data_non_default_context(self):
    context = ir.Context()
    mytype = context.get_identified_type('MyType')
    mytype.elements = [ir.IntType(32)]
    td = llvm.create_target_data('e-m:e-i64:64-f80:128-n8:16:32:64-S128')
    self.assertEqual(mytype.get_abi_size(td, context=context), 4)