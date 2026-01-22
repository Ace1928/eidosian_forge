from __future__ import annotations
import re
import sys
import traceback
from twisted.python.compat import nativeString
from twisted.trial import unittest
from twisted.web import error
from twisted.web.template import Tag
def test_reprWithRootsAndWithTraceback(self) -> None:
    """
        The representation of a L{FlattenerError} initialized with roots and a
        traceback contains a formatted representation of those roots (using
        C{_formatRoot}) and a formatted traceback.
        """
    e = self.makeFlattenerError(['a', 'b'])
    e._formatRoot = self.fakeFormatRoot
    self.assertTrue(re.match('Exception while flattening:\n  R\\(a\\)\n  R\\(b\\)\n  File "[^"]*", line [0-9]*, in makeFlattenerError\n    raise RuntimeError\\("oh noes"\\)\nRuntimeError: oh noes\n$', repr(e), re.M | re.S), repr(e))