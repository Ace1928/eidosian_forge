import warnings
from twisted.trial.unittest import TestCase
def test_orderedDifferentContainers_gt(self):
    """
        L{twisted.python.constants._Constant.__gt__} returns C{NotImplemented}
        when comparing constants belonging to different containers.
        """
    self.assertEqual(NotImplemented, NamedLetters.alpha.__gt__(MoreNamedLetters.digamma))