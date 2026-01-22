from __future__ import annotations
from typing import Sequence
from twisted.trial.unittest import TestCase
from twisted.web.http_headers import Headers
from twisted.web.test.requesthelper import (
def test_addRawHeaderTypeCheckValue(self) -> None:
    """
        L{Headers.addRawHeader} requires value to be a L{bytes} or L{str}
        string.
        """
    h = Headers()
    e = self.assertRaises(TypeError, h.addRawHeader, b'key', None)
    self.assertEqual(e.args[0], "Header value is an instance of <class 'NoneType'>, not bytes or str")