from twisted.trial import unittest
from twisted.words.protocols.jabber import error
from twisted.words.xish import domish
def test_appCondition(self) -> None:
    """
        Test parsing of an error element with an app specific condition.
        """
    condition = self.error.addElement(('testns', 'condition'))
    result = error._parseError(self.error, 'errorns')
    self.assertEqual(condition, result['appCondition'])