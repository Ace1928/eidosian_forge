import warnings
from twisted.trial.unittest import TestCase
def test_orderedDifferentContainers_ge(self):
    """
        L{twisted.python.constants._Constant.__ge__} returns C{NotImplemented}
        when comparing constants belonging to different containers.
        """
    self.assertEqual(NotImplemented, NamedLetters.alpha.__ge__(MoreNamedLetters.digamma))