from twisted.python import usage
from twisted.trial import unittest
def test_underscoreOption(self):
    """
        A dash in an option name is translated to an underscore before being
        dispatched to a handler.
        """
    self.usage.parseOptions(['--under-score', 'foo'])
    self.assertEqual(self.usage.underscoreValue, 'foo')