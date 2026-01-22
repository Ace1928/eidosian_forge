import warnings
from twisted.trial.unittest import TestCase
def test_constantsDeprecation(self):
    """
        L{twisted.python.constants} is deprecated since Twisted 16.5.
        """
    from twisted.python import constants
    constants
    warningsShown = self.flushWarnings([self.test_constantsDeprecation])
    self.assertEqual(1, len(warningsShown))
    self.assertEqual('twisted.python.constants was deprecated in Twisted 16.5.0: Please use constantly from PyPI instead.', warningsShown[0]['message'])