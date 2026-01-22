from __future__ import annotations
from typing import Iterable
from typing_extensions import Protocol
from twisted.trial.unittest import SynchronousTestCase
from ..url import URL
def test_querySet(self) -> None:
    """
        L{URL.set} replaces query parameters by name.
        """
    urlpath = URL.fromText(theurl)
    self.assertEqual('http://www.foo.com/a/nice/path/?zot=32&zut', urlpath.set('zot', '32').asText())
    self.assertEqual('http://www.foo.com/a/nice/path/?zot&zut=itworked', urlpath.set('zot').set('zut', 'itworked').asText())
    self.assertEqual('http://www.foo.com/a/nice/path/?zot=32&zut', urlpath.add('zot', 'xxx').set('zot', '32').asText())