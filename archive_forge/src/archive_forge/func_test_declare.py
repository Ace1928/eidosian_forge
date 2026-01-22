import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def test_declare(self):
    func = self.function()
    asm = self.descr(func).strip()
    self.assertEqual(asm.strip(), 'declare %s' % self.proto)