import ast
import io
import sys
import traceback
import testtools
from testtools.compat import (
from testtools.matchers import (
def test_encoding_as_none_becomes_ascii(self):
    """A stream with encoding value of None gets ascii/replace strings"""
    sout = _FakeOutputStream()
    sout.encoding = None
    unicode_output_stream(sout).write(self.uni)
    self.assertEqual([_b('pa???n')], sout.writelog)