from urllib.parse import urlparse
from twisted.trial import unittest
from twisted.web import client
def test_originForm(self):
    """
        L{client.URI.originForm} produces an absolute I{URI} path including
        the I{URI} path.
        """
    uri = client.URI.fromBytes(self.makeURIString(b'http://HOST/foo'))
    self.assertEqual(b'/foo', uri.originForm)