from twisted.python import usage
from twisted.trial import unittest
def test_nonCallable(self):
    """
        Using a non-callable type fails.
        """
    us = WrongTypedOptions()
    argV = '--barwrong egg'.split()
    self.assertRaises(TypeError, us.parseOptions, argV)