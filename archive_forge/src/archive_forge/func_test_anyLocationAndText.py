from twisted.trial import unittest
from twisted.words.xish import xpath
from twisted.words.xish.domish import Element
from twisted.words.xish.xpath import XPathQuery
from twisted.words.xish.xpathparser import SyntaxError  # type: ignore[attr-defined]
def test_anyLocationAndText(self) -> None:
    """
        Test finding any nodes named gar and getting their text contents.
        """
    xp = XPathQuery('//gar')
    self.assertEqual(xp.matches(self.e), True)
    self.assertEqual(xp.queryForNodes(self.e), [self.gar1, self.gar2, self.gar3, self.gar4])
    self.assertEqual(xp.queryForStringList(self.e), ['DEF', 'ABC', 'JKL', 'MNO'])