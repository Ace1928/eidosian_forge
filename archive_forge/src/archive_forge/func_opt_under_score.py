from twisted.python import usage
from twisted.trial import unittest
def opt_under_score(self, value):
    """
        This option has an underscore in its name to exercise the _ to -
        translation.
        """
    self.underscoreValue = value