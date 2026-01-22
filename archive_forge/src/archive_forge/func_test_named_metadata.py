import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def test_named_metadata(self):
    mod = self.module()
    m0 = mod.add_metadata([int32(123)])
    m1 = mod.add_metadata([int64(456)])
    nmd = mod.add_named_metadata('foo')
    nmd.add(m0)
    nmd.add(m1)
    nmd.add(m0)
    self.assert_ir_line('!foo = !{ !0, !1, !0 }', mod)
    self.assert_valid_ir(mod)
    self.assertIs(nmd, mod.get_named_metadata('foo'))
    with self.assertRaises(KeyError):
        mod.get_named_metadata('bar')