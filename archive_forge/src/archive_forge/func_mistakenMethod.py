from twisted.internet.defer import (
from twisted.trial.unittest import SynchronousTestCase, TestCase
def mistakenMethod(self):
    """
        This method mistakenly invokes L{returnValue}, despite the fact that it
        is not decorated with L{inlineCallbacks}.
        """
    returnValue(1)