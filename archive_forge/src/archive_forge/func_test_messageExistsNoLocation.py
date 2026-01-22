from __future__ import annotations
import re
import sys
import traceback
from twisted.python.compat import nativeString
from twisted.trial import unittest
from twisted.web import error
from twisted.web.template import Tag
def test_messageExistsNoLocation(self) -> None:
    """
        If a C{message} argument is passed to the L{InfiniteRedirection}
        constructor and no location is provided, C{message} doesn't try to
        include the empty location.
        """
    e = error.InfiniteRedirection(b'200', b'My own message')
    self.assertEqual(e.message, b'My own message')