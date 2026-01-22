import ast
import io
import sys
import traceback
import testtools
from testtools.compat import (
from testtools.matchers import (
def test_bytes_examples_multiline(self):
    for b, _, expected in self.bytes_examples:
        actual = text_repr(b, multiline=True)
        self.assertEqual(actual, self.b_prefix + expected)
        self.assertEqual(ast.literal_eval(actual), b)