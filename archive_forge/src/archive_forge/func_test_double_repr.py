import math
import sys
import unittest
from llvmlite.ir import (
from llvmlite.tests import TestCase
def test_double_repr(self):

    def check_repr(val, expected):
        c = Constant(DoubleType(), val)
        self.assertEqual(str(c), expected)
    check_repr(math.pi, 'double 0x400921fb54442d18')
    check_repr(float('inf'), 'double 0x7ff0000000000000')
    check_repr(float('-inf'), 'double 0xfff0000000000000')