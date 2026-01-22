from twisted.trial import unittest
from twisted.words.xish import xpath
from twisted.words.xish.domish import Element
from twisted.words.xish.xpath import XPathQuery
from twisted.words.xish.xpathparser import SyntaxError  # type: ignore[attr-defined]
def test_booleanOperatorsParens(self) -> None:
    """
        Test multiple boolean operators in condition with parens.
        """
    xp = XPathQuery("//bar[@attrib4='value4' and\n                                 (@attrib5='value4' or @attrib5='value6')]")
    self.assertEqual(xp.matches(self.e), True)
    self.assertEqual(xp.queryForNodes(self.e), [self.bar6, self.bar7])