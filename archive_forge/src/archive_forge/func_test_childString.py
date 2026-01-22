from twisted.python import urlpath
from twisted.trial import unittest
def test_childString(self):
    """
        Calling C{str()} with a C{URLPath.child()} will return a URL which is
        the child of the URL it was instantiated with.
        """
    self.assertEqual(str(self.path.child(b'hello')), 'http://example.com/foo/bar/hello')
    self.assertEqual(str(self.path.child(b'hello').child(b'')), 'http://example.com/foo/bar/hello/')
    self.assertEqual(str(self.path.child(b'hello', keepQuery=True)), 'http://example.com/foo/bar/hello?yes=no&no=yes')