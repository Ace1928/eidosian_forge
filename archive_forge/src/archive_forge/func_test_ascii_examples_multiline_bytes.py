import ast
import io
import sys
import traceback
import testtools
from testtools.compat import (
from testtools.matchers import (
def test_ascii_examples_multiline_bytes(self):
    for s, _, expected in self.ascii_examples:
        b = _b(s)
        actual = text_repr(b, multiline=True)
        self.assertEqual(actual, self.b_prefix + expected)
        self.assertEqual(ast.literal_eval(actual), b)