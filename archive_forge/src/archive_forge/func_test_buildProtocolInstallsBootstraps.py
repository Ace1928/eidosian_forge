from twisted.internet import protocol
from twisted.python import failure
from twisted.trial import unittest
from twisted.words.xish import domish, utility, xmlstream
def test_buildProtocolInstallsBootstraps(self):
    """
        The protocol factory installs bootstrap event handlers on the protocol.
        """
    called = []

    def cb(data):
        called.append(data)
    self.factory.addBootstrap('//event/myevent', cb)
    xs = self.factory.buildProtocol(None)
    xs.dispatch(None, '//event/myevent')
    self.assertEqual(1, len(called))