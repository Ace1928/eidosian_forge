from twisted.trial import unittest
from twisted.words.xish import xpath
from twisted.words.xish.domish import Element
from twisted.words.xish.xpath import XPathQuery
from twisted.words.xish.xpathparser import SyntaxError  # type: ignore[attr-defined]
def test_locationFooBar(self) -> None:
    """
        Test matching foo with child bar.
        """
    xp = XPathQuery('/foo/bar')
    self.assertEqual(xp.matches(self.e), 1)