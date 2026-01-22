from __future__ import annotations
from typing import Sequence
from twisted.trial.unittest import TestCase
from twisted.web.http_headers import Headers
from twisted.web.test.requesthelper import (
def test_subclassRepr(self) -> None:
    """
        The L{repr} of an instance of a subclass of L{Headers} uses the name
        of the subclass instead of the string C{"Headers"}.
        """
    foo = 'fooá'
    bar = 'bar☃'
    baz = 'baz'
    fooEncoded = "b'foo\\xe1'"
    barEncoded = "b'bar\\xe2\\x98\\x83'"

    class FunnyHeaders(Headers):
        pass
    self.assertEqual(repr(FunnyHeaders({foo: [bar, baz]})), 'FunnyHeaders({%s: [%s, %r]})' % (fooEncoded, barEncoded, baz.encode('utf8')))