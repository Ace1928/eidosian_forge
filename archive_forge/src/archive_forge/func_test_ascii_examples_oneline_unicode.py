import ast
import io
import sys
import traceback
import testtools
from testtools.compat import (
from testtools.matchers import (
def test_ascii_examples_oneline_unicode(self):
    for s, expected, _ in self.ascii_examples:
        u = s
        actual = text_repr(u, multiline=False)
        self.assertEqual(actual, self.u_prefix + expected)
        self.assertEqual(ast.literal_eval(actual), u)