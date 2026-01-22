import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def check_func_body(self, func, asm):
    expected = self._normalize_asm(asm)
    actual = self.descr(func)
    actual = actual.partition('{')[2].rpartition('}')[0]
    self.assertEqual(actual.strip(), expected.strip())