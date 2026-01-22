import warnings
from twisted.trial.unittest import TestCase
def test_lookupDuplicateByValue(self):
    """
        If more than one constant is associated with a particular value,
        L{Flags.lookupByValue} returns whichever of them is defined first.
        """

    class TIMEX(Flags):
        ADJ_OFFSET = FlagConstant(1)
        MOD_OFFSET = FlagConstant(1)
    self.assertIs(TIMEX.lookupByValue(1), TIMEX.ADJ_OFFSET)