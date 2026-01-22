from __future__ import annotations
from typing import Sequence
from twisted.trial.unittest import TestCase
from twisted.web.http_headers import Headers
from twisted.web.test.requesthelper import (
def test_getAllRawHeaders(self) -> None:
    """
        L{Headers.getAllRawHeaders} returns an iterable of (k, v) pairs, where
        C{k} is the canonicalized representation of the header name, and C{v}
        is a sequence of values.
        """
    h = Headers()
    h.setRawHeaders('testá', ['lemurs'])
    h.setRawHeaders('www-authenticate', ['basic aksljdlk='])
    h.setRawHeaders('content-md5', ['kjdfdfgdfgnsd'])
    allHeaders = {(k, tuple(v)) for k, v in h.getAllRawHeaders()}
    self.assertEqual(allHeaders, {(b'WWW-Authenticate', (b'basic aksljdlk=',)), (b'Content-MD5', (b'kjdfdfgdfgnsd',)), (b'Test\xe1', (b'lemurs',))})