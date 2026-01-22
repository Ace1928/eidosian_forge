from twisted.trial import unittest
from twisted.words.xish import xpath
from twisted.words.xish.domish import Element
from twisted.words.xish.xpath import XPathQuery
from twisted.words.xish.xpathparser import SyntaxError  # type: ignore[attr-defined]
def test_anyLocation(self) -> None:
    """
        Test finding any nodes named bar.
        """
    xp = XPathQuery('//bar')
    self.assertEqual(xp.matches(self.e), True)
    self.assertEqual(xp.queryForNodes(self.e), [self.bar1, self.bar2, self.bar3, self.bar4, self.bar5, self.bar6, self.bar7])