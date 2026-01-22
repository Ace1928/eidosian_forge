from __future__ import annotations
from typing import Sequence
from twisted.trial.unittest import TestCase
from twisted.web.http_headers import Headers
from twisted.web.test.requesthelper import (
def test_rawHeadersTypeCheckingName(self) -> None:
    """
        L{Headers.setRawHeaders} requires C{name} to be a L{bytes} or
        L{str} string.
        """
    h = Headers()
    e = self.assertRaises(TypeError, h.setRawHeaders, None, [b'foo'])
    self.assertEqual(e.args[0], "Header name is an instance of <class 'NoneType'>, not bytes or str")