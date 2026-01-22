import warnings
from twisted.trial.unittest import TestCase
def test_asForeignClassAttribute(self):
    """
        A constant defined on a L{Names} subclass may be set as an attribute of
        another class and then retrieved using that attribute.
        """

    class Another:
        something = self.METHOD.GET
    self.assertIs(self.METHOD.GET, Another.something)