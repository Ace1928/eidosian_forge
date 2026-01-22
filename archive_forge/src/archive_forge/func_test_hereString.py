from twisted.python import urlpath
from twisted.trial import unittest
def test_hereString(self):
    """
        Calling C{str()} with a C{URLPath.here()} will return a URL which is
        the URL that it was instantiated with, without any file, query, or
        fragment.
        """
    self.assertEqual(str(self.path.here()), 'http://example.com/foo/')
    self.assertEqual(str(self.path.here(keepQuery=True)), 'http://example.com/foo/?yes=no&no=yes')
    self.assertEqual(str(self.path.child(b'').here()), 'http://example.com/foo/bar/')