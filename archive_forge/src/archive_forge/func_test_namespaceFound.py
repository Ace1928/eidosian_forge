from twisted.trial import unittest
from twisted.words.xish import xpath
from twisted.words.xish.domish import Element
from twisted.words.xish.xpath import XPathQuery
from twisted.words.xish.xpathparser import SyntaxError  # type: ignore[attr-defined]
def test_namespaceFound(self) -> None:
    """
        Test matching node with namespace.
        """
    xp = XPathQuery("/foo[@xmlns='testns']/bar")
    self.assertEqual(xp.matches(self.e), 1)