from __future__ import absolute_import
import sys
import os.path
from textwrap import dedent
from contextlib import contextmanager
from tempfile import NamedTemporaryFile
from Cython.Compiler.ParseTreeTransforms import NormalizeTree, InterpretCompilerDirectives
from Cython.Compiler import Main, Symtab, Visitor, Options
from Cython.TestUtils import TransformTest
def test_conflicting_types_in_function(self):
    code = "        def func(a, b):\n            print(a)\n            a = 1\n            b += a\n            a = 'abc'\n            return a, str(b)\n\n        print(func(1.5, 2))\n        "
    types = self._test(code)
    self.assertIn(('func', (1, 0)), types)
    variables = types.pop(('func', (1, 0)))
    self.assertFalse(types)
    self.assertEqual({'a': set(['float', 'int', 'str']), 'b': set(['int'])}, variables)