from twisted.internet import protocol
from twisted.python import failure
from twisted.trial import unittest
from twisted.words.xish import domish, utility, xmlstream
def test_installBootstraps(self):
    """
        Dispatching an event fires registered bootstrap observers.
        """
    called = []

    def cb(data):
        called.append(data)
    dispatcher = DummyProtocol()
    self.factory.addBootstrap('//event/myevent', cb)
    self.factory.installBootstraps(dispatcher)
    dispatcher.dispatch(None, '//event/myevent')
    self.assertEqual(1, len(called))