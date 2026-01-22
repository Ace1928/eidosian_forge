import warnings
from twisted.trial.unittest import TestCase
def test_orderedDifferentConstants_le(self):
    """
        L{twisted.python.constants._Constant.__le__} returns C{NotImplemented}
        when comparing constants of different types.
        """
    self.assertEqual(NotImplemented, NamedLetters.alpha.__le__(ValuedLetters.alpha))