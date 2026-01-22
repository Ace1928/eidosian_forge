import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def test_debug_info_unicode_string(self):
    mod = self.module()
    mod.add_debug_info('DILocalVariable', {'name': 'a∆'})
    strmod = str(mod)
    name = ''.join(map(lambda x: f'\\{x:02x}', '∆'.encode()))
    self.assert_ir_line(f'!0 = !DILocalVariable(name: "a{name}")', strmod)