from twisted.internet import defer, error, interfaces, reactor, task
from twisted.internet.main import installReactor
from twisted.internet.test.modulehelpers import NoReactor
from twisted.trial import unittest
from twisted.python import failure
def test_reprFunction(self):
    """
        L{LoopingCall.__repr__} includes the wrapped function's name.
        """
    self.assertEqual(repr(task.LoopingCall(installReactor, 1, key=2)), "LoopingCall<None>(installReactor, *(1,), **{'key': 2})")