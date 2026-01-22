import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def test_abi_size(self):
    td = llvm.create_target_data('e-m:e-i64:64-f80:128-n8:16:32:64-S128')

    def check(tp, expected):
        self.assertEqual(tp.get_abi_size(td), expected)
    check(int8, 1)
    check(int32, 4)
    check(int64, 8)
    check(ir.ArrayType(int8, 5), 5)
    check(ir.ArrayType(int32, 5), 20)
    check(ir.LiteralStructType((dbl, flt, flt)), 16)