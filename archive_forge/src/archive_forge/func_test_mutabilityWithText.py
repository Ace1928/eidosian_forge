from twisted.python import urlpath
from twisted.trial import unittest
def test_mutabilityWithText(self, stringType=str):
    """
        Setting attributes on L{urlpath.URLPath} should change the value
        returned by L{str}.

        @param stringType: a callable to parameterize this test for different
            text types.
        @type stringType: 1-argument callable taking L{str} and returning
            L{str} or L{bytes}.
        """
    self.path.scheme = stringType('https')
    self.assertEqual(str(self.path), 'https://example.com/foo/bar?yes=no&no=yes#footer')
    self.path.netloc = stringType('another.example.invalid')
    self.assertEqual(str(self.path), 'https://another.example.invalid/foo/bar?yes=no&no=yes#footer')
    self.path.path = stringType('/hello')
    self.assertEqual(str(self.path), 'https://another.example.invalid/hello?yes=no&no=yes#footer')
    self.path.query = stringType('alpha=omega&opposites=same')
    self.assertEqual(str(self.path), 'https://another.example.invalid/hello?alpha=omega&opposites=same#footer')
    self.path.fragment = stringType('header')
    self.assertEqual(str(self.path), 'https://another.example.invalid/hello?alpha=omega&opposites=same#header')