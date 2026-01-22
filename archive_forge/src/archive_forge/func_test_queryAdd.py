from __future__ import annotations
from typing import Iterable
from typing_extensions import Protocol
from twisted.trial.unittest import SynchronousTestCase
from ..url import URL
def test_queryAdd(self) -> None:
    """
        L{URL.add} adds query parameters.
        """
    self.assertEqual('http://www.foo.com/a/nice/path/?foo=bar', URL.fromText('http://www.foo.com/a/nice/path/').add('foo', 'bar').asText())
    self.assertEqual('http://www.foo.com/?foo=bar', URL(host='www.foo.com').add('foo', 'bar').asText())
    urlpath = URL.fromText(theurl)
    self.assertEqual('http://www.foo.com/a/nice/path/?zot=23&zut&burp', urlpath.add('burp').asText())
    self.assertEqual('http://www.foo.com/a/nice/path/?zot=23&zut&burp=xxx', urlpath.add('burp', 'xxx').asText())
    self.assertEqual('http://www.foo.com/a/nice/path/?zot=23&zut&burp=xxx&zing', urlpath.add('burp', 'xxx').add('zing').asText())
    self.assertEqual('http://www.foo.com/a/nice/path/?zot=23&zut&zing&burp=xxx', urlpath.add('zing').add('burp', 'xxx').asText())
    self.assertEqual('http://www.foo.com/a/nice/path/?zot=23&zut&burp=xxx&zot=32', urlpath.add('burp', 'xxx').add('zot', '32').asText())