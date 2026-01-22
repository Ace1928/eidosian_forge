import ast
import io
import sys
import traceback
import testtools
from testtools.compat import (
from testtools.matchers import (
def test_ascii_examples_multiline_unicode(self):
    for s, _, expected in self.ascii_examples:
        actual = text_repr(s, multiline=True)
        self.assertEqual(actual, self.u_prefix + expected)
        self.assertEqual(ast.literal_eval(actual), s)