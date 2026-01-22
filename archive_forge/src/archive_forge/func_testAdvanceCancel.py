from twisted.internet import defer, error, interfaces, reactor, task
from twisted.internet.main import installReactor
from twisted.internet.test.modulehelpers import NoReactor
from twisted.trial import unittest
from twisted.python import failure
def testAdvanceCancel(self):
    """
        Test attempting to cancel the call in a callback.

        AlreadyCalled should be raised, not for example a ValueError from
        removing the call from Clock.calls. This requires call.called to be
        set before the callback is called.
        """
    c = task.Clock()

    def cb():
        self.assertRaises(error.AlreadyCalled, call.cancel)
    call = c.callLater(1, cb)
    c.advance(1)