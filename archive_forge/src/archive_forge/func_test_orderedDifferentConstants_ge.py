import warnings
from twisted.trial.unittest import TestCase
def test_orderedDifferentConstants_ge(self):
    """
        L{twisted.python.constants._Constant.__ge__} returns C{NotImplemented}
        when comparing constants of different types.
        """
    self.assertEqual(NotImplemented, NamedLetters.alpha.__ge__(ValuedLetters.alpha))