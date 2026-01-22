from twisted.internet import protocol
from twisted.python import failure
from twisted.trial import unittest
from twisted.words.xish import domish, utility, xmlstream
def test_addAndRemoveBootstrap(self):
    """
        Test addition and removal of a bootstrap event handler.
        """
    called = []

    def cb(data):
        called.append(data)
    self.factory.addBootstrap('//event/myevent', cb)
    self.factory.removeBootstrap('//event/myevent', cb)
    dispatcher = DummyProtocol()
    self.factory.installBootstraps(dispatcher)
    dispatcher.dispatch(None, '//event/myevent')
    self.assertFalse(called)