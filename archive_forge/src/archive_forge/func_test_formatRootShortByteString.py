from __future__ import annotations
import re
import sys
import traceback
from twisted.python.compat import nativeString
from twisted.trial import unittest
from twisted.web import error
from twisted.web.template import Tag
def test_formatRootShortByteString(self) -> None:
    """
        The C{_formatRoot} method formats a short byte string using the
        built-in repr.
        """
    e = self.makeFlattenerError()
    self.assertEqual(e._formatRoot(b'abcd'), repr(b'abcd'))