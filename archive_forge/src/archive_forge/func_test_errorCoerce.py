from twisted.python import usage
from twisted.trial import unittest
def test_errorCoerce(self):
    """
        Test error path.
        """
    self.assertRaises(ValueError, usage.portCoerce, '')
    self.assertRaises(ValueError, usage.portCoerce, '-21')
    self.assertRaises(ValueError, usage.portCoerce, '212189')
    self.assertRaises(ValueError, usage.portCoerce, 'foo')