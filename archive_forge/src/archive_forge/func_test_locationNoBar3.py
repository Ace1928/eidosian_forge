from twisted.trial import unittest
from twisted.words.xish import xpath
from twisted.words.xish.domish import Element
from twisted.words.xish.xpath import XPathQuery
from twisted.words.xish.xpathparser import SyntaxError  # type: ignore[attr-defined]
def test_locationNoBar3(self) -> None:
    """
        Test not finding bar3.
        """
    xp = XPathQuery('/foo/bar3')
    self.assertEqual(xp.matches(self.e), 0)