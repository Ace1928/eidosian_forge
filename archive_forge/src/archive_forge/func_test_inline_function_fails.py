from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import ast
import textwrap
import unittest
from pasta.augment import inline
from pasta.base import test_utils
def test_inline_function_fails(self):
    src = 'def func(): pass\nfunc()\n'
    t = ast.parse(src)
    with self.assertRaisesRegexp(inline.InlineError, "'func' is not a constant; it has type %r" % ast.FunctionDef):
        inline.inline_name(t, 'func')