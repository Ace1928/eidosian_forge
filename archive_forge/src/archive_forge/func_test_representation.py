import warnings
from twisted.trial.unittest import TestCase
def test_representation(self):
    """
        The string representation of a L{FlagConstant} instance which results
        from C{~} includes the names of all the flags which were not set in the
        original constant.
        """
    flag = ~self.FXF.WRITE
    self.assertEqual('<FXF={APPEND,EXCLUSIVE,READ,TEXT}>', repr(flag))