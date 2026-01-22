from __future__ import annotations
import re
import sys
import traceback
from twisted.python.compat import nativeString
from twisted.trial import unittest
from twisted.web import error
from twisted.web.template import Tag
def test_messageExistsLocationExists(self) -> None:
    """
        If a C{message} argument is passed to the L{InfiniteRedirection}
        constructor, the C{message} isn't affected by the value of C{status}.
        """
    e = error.InfiniteRedirection(b'200', b'My own message', location=b'/foo')
    self.assertEqual(e.message, b'My own message to /foo')