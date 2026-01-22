import warnings
from twisted.trial.unittest import TestCase
def test_nonequality(self):
    """
        Two different L{NamedConstant} instances do not compare equal to each
        other.
        """
    first = NamedConstant()
    first._realize(self.container, 'bar', None)
    second = NamedConstant()
    second._realize(self.container, 'bar', None)
    self.assertFalse(first == second)
    self.assertTrue(first != second)