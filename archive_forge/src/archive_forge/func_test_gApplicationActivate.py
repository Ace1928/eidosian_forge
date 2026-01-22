from __future__ import annotations
from unittest import skipIf
from twisted.internet.error import ReactorAlreadyRunning
from twisted.internet.test.reactormixins import ReactorBuilder
from twisted.trial.unittest import SkipTest, TestCase
def test_gApplicationActivate(self) -> None:
    """
        L{Gio.Application} instances can be registered with a gireactor.
        """
    self.reactorFactory = lambda: gireactor.GIReactor(useGtk=False)
    reactor = self.buildReactor()
    app = Gio.Application(application_id='com.twistedmatrix.trial.gireactor', flags=Gio.ApplicationFlags.FLAGS_NONE)
    self.runReactor(app, reactor)