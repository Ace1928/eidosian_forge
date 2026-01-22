import warnings
from twisted.trial.unittest import TestCase
def test_orderedDifferentContainers_le(self):
    """
        L{twisted.python.constants._Constant.__le__} returns C{NotImplemented}
        when comparing constants belonging to different containers.
        """
    self.assertEqual(NotImplemented, NamedLetters.alpha.__le__(MoreNamedLetters.digamma))