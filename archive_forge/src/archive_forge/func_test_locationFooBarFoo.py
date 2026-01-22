from twisted.trial import unittest
from twisted.words.xish import xpath
from twisted.words.xish.domish import Element
from twisted.words.xish.xpath import XPathQuery
from twisted.words.xish.xpathparser import SyntaxError  # type: ignore[attr-defined]
def test_locationFooBarFoo(self) -> None:
    """
        Test finding foos at the second level.
        """
    xp = XPathQuery('/foo/bar/foo')
    self.assertEqual(xp.matches(self.e), 1)
    self.assertEqual(xp.queryForNodes(self.e), [self.subfoo, self.subfoo3, self.subfoo4])