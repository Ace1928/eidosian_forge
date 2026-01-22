from urllib.parse import urlparse
from twisted.trial import unittest
from twisted.web import client
def test_param(self):
    """
        Parse I{URI} parameters from a I{URI}.
        """
    uri = self.makeURIString(b'http://HOST/foo/bar;param')
    parsed = client.URI.fromBytes(uri)
    self.assertURIEquals(parsed, scheme=b'http', netloc=self.uriHost, host=self.host, port=80, path=b'/foo/bar', params=b'param')
    self.assertEqual(uri, parsed.toBytes())