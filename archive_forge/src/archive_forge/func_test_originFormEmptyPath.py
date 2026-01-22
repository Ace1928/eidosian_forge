from urllib.parse import urlparse
from twisted.trial import unittest
from twisted.web import client
def test_originFormEmptyPath(self):
    """
        L{client.URI.originForm} produces a path of C{b'/'} when the I{URI}
        specifies an empty path.
        """
    uri = client.URI.fromBytes(self.makeURIString(b'http://HOST/'))
    self.assertEqual(b'/', uri.originForm)