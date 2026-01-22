from twisted.trial import unittest
from twisted.words.xish import xpath
from twisted.words.xish.domish import Element
from twisted.words.xish.xpath import XPathQuery
from twisted.words.xish.xpathparser import SyntaxError  # type: ignore[attr-defined]
def test_booleanOperatorsNoParens(self) -> None:
    """
        Test multiple boolean operators in condition without parens.
        """
    xp = XPathQuery("//bar[@attrib5='value4' or\n                                 @attrib5='value5' or\n                                 @attrib5='value6']")
    self.assertEqual(xp.matches(self.e), True)
    self.assertEqual(xp.queryForNodes(self.e), [self.bar5, self.bar6, self.bar7])