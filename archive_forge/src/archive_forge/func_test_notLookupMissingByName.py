import warnings
from twisted.trial.unittest import TestCase
def test_notLookupMissingByName(self):
    """
        Names not defined with a L{FlagConstant} instance cannot be looked up
        using L{Flags.lookupByName}.
        """
    self.assertRaises(ValueError, self.FXF.lookupByName, 'lookupByName')
    self.assertRaises(ValueError, self.FXF.lookupByName, '__init__')
    self.assertRaises(ValueError, self.FXF.lookupByName, 'foo')