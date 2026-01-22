from __future__ import absolute_import
import sys
import os.path
from textwrap import dedent
from contextlib import contextmanager
from tempfile import NamedTemporaryFile
from Cython.Compiler.ParseTreeTransforms import NormalizeTree, InterpretCompilerDirectives
from Cython.Compiler import Main, Symtab, Visitor, Options
from Cython.TestUtils import TransformTest
def test_typing_function_dict(self):
    code = "        def func(x):\n            a = dict()\n            b = {i: i**2 for i in range(10)}\n            c = x\n\n        print(func({1:2, 'x':7}))\n        "
    types = self._test(code)
    self.assertIn(('func', (1, 0)), types)
    variables = types.pop(('func', (1, 0)))
    self.assertFalse(types)
    self.assertEqual({'a': set(['dict']), 'b': set(['dict']), 'c': set(['dict']), 'x': set(['dict'])}, variables)