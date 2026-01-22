from __future__ import annotations
from typing import Sequence
from twisted.trial.unittest import TestCase
from twisted.web.http_headers import Headers
from twisted.web.test.requesthelper import (
def test_reprWithRawBytes(self) -> None:
    """
        The L{repr} of a L{Headers} instance shows the names and values of all
        the headers it contains, not attempting to decode any raw bytes.
        """
    foo = b'foo'
    bar = b'bar\xe1'
    baz = b'baz\xe1'
    self.assertEqual(repr(Headers({foo: [bar, baz]})), f'Headers({{{foo!r}: [{bar!r}, {baz!r}]}})')