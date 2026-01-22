import itertools
from twisted.positioning import _sentence
from twisted.trial.unittest import TestCase
def test_knownTypeRepr(self):
    """
        Test the repr of an empty sentence of known type.
        """
    sentence = self.sentenceClass({'type': self.sentenceType})
    expectedRepr = self._expectedRepr(self.sentenceType)
    self.assertEqual(repr(sentence), expectedRepr)