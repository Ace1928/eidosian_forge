from __future__ import annotations
import re
import sys
import traceback
from twisted.python.compat import nativeString
from twisted.trial import unittest
from twisted.web import error
from twisted.web.template import Tag
def test_formatRootLongUnicodeString(self) -> None:
    """
        The C{_formatRoot} method formats a long unicode string using the
        built-in repr with an ellipsis.
        """
    e = self.makeFlattenerError()
    longString = nativeString('abcde-' * 20)
    self.assertEqual(e._formatRoot(longString), repr('abcde-abcde-abcde-ab<...>e-abcde-abcde-abcde-'))