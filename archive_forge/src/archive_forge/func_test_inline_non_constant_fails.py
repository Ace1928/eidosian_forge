from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import ast
import textwrap
import unittest
from pasta.augment import inline
from pasta.base import test_utils
def test_inline_non_constant_fails(self):
    src = textwrap.dedent('        NOT_A_CONSTANT = "foo"\n        NOT_A_CONSTANT += "bar"\n        ')
    t = ast.parse(src)
    with self.assertRaisesRegexp(inline.InlineError, "'NOT_A_CONSTANT' is not a constant"):
        inline.inline_name(t, 'NOT_A_CONSTANT')