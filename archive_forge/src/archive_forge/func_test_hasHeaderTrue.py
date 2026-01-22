from __future__ import annotations
from typing import Sequence
from twisted.trial.unittest import TestCase
from twisted.web.http_headers import Headers
from twisted.web.test.requesthelper import (
def test_hasHeaderTrue(self) -> None:
    """
        Check that L{Headers.hasHeader} returns C{True} when the given header
        is found.
        """
    h = Headers()
    h.setRawHeaders('testá', ['lemur'])
    self.assertTrue(h.hasHeader('testá'))
    self.assertTrue(h.hasHeader('Testá'))
    self.assertTrue(h.hasHeader(b'test\xe1'))
    self.assertTrue(h.hasHeader(b'Test\xe1'))