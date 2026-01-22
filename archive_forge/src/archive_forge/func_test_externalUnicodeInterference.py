from urllib.parse import urlparse
from twisted.trial import unittest
from twisted.web import client
def test_externalUnicodeInterference(self):
    """
        L{client.URI.fromBytes} parses the scheme, host, and path elements
        into L{bytes}, even when passed an URL which has previously been passed
        to L{urlparse} as a L{unicode} string.
        """
    goodInput = self.makeURIString(b'http://HOST/path')
    badInput = goodInput.decode('ascii')
    urlparse(badInput)
    uri = client.URI.fromBytes(goodInput)
    self.assertIsInstance(uri.scheme, bytes)
    self.assertIsInstance(uri.host, bytes)
    self.assertIsInstance(uri.path, bytes)