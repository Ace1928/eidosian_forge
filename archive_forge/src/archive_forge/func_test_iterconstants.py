import warnings
from twisted.trial.unittest import TestCase
def test_iterconstants(self):
    """
        L{Flags.iterconstants} returns an iterator over all of the constants
        defined in the class, in the order they were defined.
        """
    constants = list(self.FXF.iterconstants())
    self.assertEqual([self.FXF.READ, self.FXF.WRITE, self.FXF.APPEND, self.FXF.EXCLUSIVE, self.FXF.TEXT], constants)