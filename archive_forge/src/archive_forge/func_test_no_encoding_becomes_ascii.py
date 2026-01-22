import ast
import io
import sys
import traceback
import testtools
from testtools.compat import (
from testtools.matchers import (
def test_no_encoding_becomes_ascii(self):
    """A stream with no encoding attribute gets ascii/replace strings"""
    sout = _FakeOutputStream()
    unicode_output_stream(sout).write(self.uni)
    self.assertEqual([_b('pa???n')], sout.writelog)