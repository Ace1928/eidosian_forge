from urllib.parse import urlparse
from twisted.trial import unittest
from twisted.web import client
def test_fragment(self):
    """
        Parse the fragment identifier from a I{URI}.
        """
    uri = self.makeURIString(b'http://HOST/foo/bar;param?a=1&b=2#frag')
    parsed = client.URI.fromBytes(uri)
    self.assertURIEquals(parsed, scheme=b'http', netloc=self.uriHost, host=self.host, port=80, path=b'/foo/bar', params=b'param', query=b'a=1&b=2', fragment=b'frag')
    self.assertEqual(uri, parsed.toBytes())