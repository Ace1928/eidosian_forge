from urllib.parse import urlparse
from twisted.trial import unittest
from twisted.web import client
def test_originFormComplex(self):
    """
        L{client.URI.originForm} produces an absolute I{URI} path including
        the I{URI} path, parameters and query string but excludes the fragment
        identifier.
        """
    uri = client.URI.fromBytes(self.makeURIString(b'http://HOST/foo;param?a=1#frag'))
    self.assertEqual(b'/foo;param?a=1', uri.originForm)