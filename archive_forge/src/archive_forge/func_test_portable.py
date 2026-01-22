from __future__ import annotations
from unittest import skipIf
from twisted.internet.error import ReactorAlreadyRunning
from twisted.internet.test.reactormixins import ReactorBuilder
from twisted.trial.unittest import SkipTest, TestCase
def test_portable(self) -> None:
    """
        L{gireactor.PortableGIReactor} doesn't support application
        registration at this time.
        """
    self.reactorFactory = gireactor.PortableGIReactor
    reactor = self.buildReactor()
    app = Gio.Application(application_id='com.twistedmatrix.trial.gireactor', flags=Gio.ApplicationFlags.FLAGS_NONE)
    self.assertRaises(NotImplementedError, reactor.registerGApplication, app)