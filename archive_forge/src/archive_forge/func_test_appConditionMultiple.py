from twisted.trial import unittest
from twisted.words.protocols.jabber import error
from twisted.words.xish import domish
def test_appConditionMultiple(self) -> None:
    """
        Test parsing of an error element with multiple app specific conditions.
        """
    self.error.addElement(('testns', 'condition'))
    condition = self.error.addElement(('testns', 'condition2'))
    result = error._parseError(self.error, 'errorns')
    self.assertEqual(condition, result['appCondition'])