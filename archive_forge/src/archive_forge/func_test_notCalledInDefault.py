from twisted.python import usage
from twisted.trial import unittest
def test_notCalledInDefault(self):
    """
        The coerce functions are not called if no values are provided.
        """
    us = WeirdCallableOptions()
    argV = []
    us.parseOptions(argV)