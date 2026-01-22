from __future__ import annotations
import re
import sys
import traceback
from twisted.python.compat import nativeString
from twisted.trial import unittest
from twisted.web import error
from twisted.web.template import Tag
def test_reprWithoutRootsAndWithoutTraceback(self) -> None:
    """
        The representation of a L{FlattenerError} initialized without roots but
        with a traceback contains a formatted traceback but no roots.
        """
    e = error.FlattenerError(RuntimeError('oh noes'), [], None)
    self.assertTrue(re.match('Exception while flattening:\nRuntimeError: oh noes\n$', repr(e), re.M | re.S), repr(e))