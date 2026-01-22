from twisted.internet import defer, error, interfaces, reactor, task
from twisted.internet.main import installReactor
from twisted.internet.test.modulehelpers import NoReactor
from twisted.trial import unittest
from twisted.python import failure
def test_reprMethod(self):
    """
        L{LoopingCall.__repr__} includes the wrapped method's full name.
        """
    self.assertEqual(repr(task.LoopingCall(TestableLoopingCall.__init__)), 'LoopingCall<None>(TestableLoopingCall.__init__, *(), **{})')