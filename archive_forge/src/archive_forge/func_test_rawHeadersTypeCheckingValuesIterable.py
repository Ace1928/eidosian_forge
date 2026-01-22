from __future__ import annotations
from typing import Sequence
from twisted.trial.unittest import TestCase
from twisted.web.http_headers import Headers
from twisted.web.test.requesthelper import (
def test_rawHeadersTypeCheckingValuesIterable(self) -> None:
    """
        L{Headers.setRawHeaders} requires values to be of type list.
        """
    h = Headers()
    self.assertRaises(TypeError, h.setRawHeaders, b'key', {b'Foo': b'bar'})