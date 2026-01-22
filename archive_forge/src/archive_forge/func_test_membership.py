import warnings
from twisted.trial.unittest import TestCase
def test_membership(self):
    """
        A L{FlagConstant} instance which results from C{|} can be
        tested for membership.
        """
    flags = self.FXF.WRITE | self.FXF.EXCLUSIVE
    self.assertIn(self.FXF.WRITE, flags)
    self.assertNotIn(self.FXF.READ, flags)