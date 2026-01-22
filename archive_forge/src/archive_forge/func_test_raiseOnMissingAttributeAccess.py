import itertools
from twisted.positioning import _sentence
from twisted.trial.unittest import TestCase
def test_raiseOnMissingAttributeAccess(self):
    """
        Accessing a nonexistent attribute raises C{AttributeError}.
        """
    sentence = self.sentenceClass({})
    self.assertRaises(AttributeError, getattr, sentence, 'BOGUS')