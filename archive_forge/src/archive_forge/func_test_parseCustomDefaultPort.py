from urllib.parse import urlparse
from twisted.trial import unittest
from twisted.web import client
def test_parseCustomDefaultPort(self):
    """
        L{client.URI.fromBytes} accepts a C{defaultPort} parameter that
        overrides the normal default port logic.
        """
    uri = client.URI.fromBytes(self.makeURIString(b'http://HOST'), defaultPort=5144)
    self.assertEqual(5144, uri.port)
    uri = client.URI.fromBytes(self.makeURIString(b'https://HOST'), defaultPort=5144)
    self.assertEqual(5144, uri.port)