import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def check_descr_regex(self, descr, asm):
    expected = self._normalize_asm(asm)
    self.assertRegex(descr, expected)