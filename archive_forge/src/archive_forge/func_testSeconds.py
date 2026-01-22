from twisted.internet import defer, error, interfaces, reactor, task
from twisted.internet.main import installReactor
from twisted.internet.test.modulehelpers import NoReactor
from twisted.trial import unittest
from twisted.python import failure
def testSeconds(self):
    """
        Test that the C{seconds} method of the fake clock returns fake time.
        """
    c = task.Clock()
    self.assertEqual(c.seconds(), 0)