from urllib.parse import urlparse
from twisted.trial import unittest
from twisted.web import client
def test_netlocHostPort(self):
    """
        Parsing a I{URI} splits the network location component into I{host} and
        I{port}.
        """
    uri = client.URI.fromBytes(self.makeURIString(b'http://HOST:5144'))
    self.assertEqual(5144, uri.port)
    self.assertEqual(self.host, uri.host)
    self.assertEqual(self.uriHost + b':5144', uri.netloc)
    uri = client.URI.fromBytes(self.makeURIString(b'http://HOST '))
    self.assertEqual(self.uriHost, uri.netloc)