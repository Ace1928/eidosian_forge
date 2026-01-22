import warnings
from twisted.trial.unittest import TestCase
def test_attributeIterconstantsIdentity(self):
    """
        The constants returned from L{Flags.iterconstants} are identical to the
        constants accessible using attributes.
        """
    constants = list(self.FXF.iterconstants())
    self.assertIs(self.FXF.READ, constants[0])
    self.assertIs(self.FXF.WRITE, constants[1])
    self.assertIs(self.FXF.APPEND, constants[2])
    self.assertIs(self.FXF.EXCLUSIVE, constants[3])
    self.assertIs(self.FXF.TEXT, constants[4])