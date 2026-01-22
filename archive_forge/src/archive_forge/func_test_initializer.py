from __future__ import annotations
from typing import Sequence
from twisted.trial.unittest import TestCase
from twisted.web.http_headers import Headers
from twisted.web.test.requesthelper import (
def test_initializer(self) -> None:
    """
        The header values passed to L{Headers.__init__} can be retrieved via
        L{Headers.getRawHeaders}. If a L{bytes} argument is given, it returns
        L{bytes} values, and if a L{str} argument is given, it returns
        L{str} values. Both are the same header value, just encoded or
        decoded.
        """
    h = Headers({'Foo': ['bar']})
    self.assertEqual(h.getRawHeaders(b'foo'), [b'bar'])
    self.assertEqual(h.getRawHeaders('foo'), ['bar'])