from twisted.python import usage
from twisted.trial import unittest
def test_invalidValues(self):
    """
        Passing wrong values raises an error.
        """
    argV = '--fooint egg'.split()
    self.assertRaises(usage.UsageError, self.usage.parseOptions, argV)