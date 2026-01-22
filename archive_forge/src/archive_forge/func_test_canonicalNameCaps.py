from __future__ import annotations
from typing import Sequence
from twisted.trial.unittest import TestCase
from twisted.web.http_headers import Headers
from twisted.web.test.requesthelper import (
def test_canonicalNameCaps(self) -> None:
    """
        L{Headers._canonicalNameCaps} returns the canonical capitalization for
        the given header.
        """
    h = Headers()
    self.assertEqual(h._canonicalNameCaps(b'test'), b'Test')
    self.assertEqual(h._canonicalNameCaps(b'test-stuff'), b'Test-Stuff')
    self.assertEqual(h._canonicalNameCaps(b'content-md5'), b'Content-MD5')
    self.assertEqual(h._canonicalNameCaps(b'dnt'), b'DNT')
    self.assertEqual(h._canonicalNameCaps(b'etag'), b'ETag')
    self.assertEqual(h._canonicalNameCaps(b'p3p'), b'P3P')
    self.assertEqual(h._canonicalNameCaps(b'te'), b'TE')
    self.assertEqual(h._canonicalNameCaps(b'www-authenticate'), b'WWW-Authenticate')
    self.assertEqual(h._canonicalNameCaps(b'x-xss-protection'), b'X-XSS-Protection')