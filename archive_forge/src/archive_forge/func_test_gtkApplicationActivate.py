from __future__ import annotations
from unittest import skipIf
from twisted.internet.error import ReactorAlreadyRunning
from twisted.internet.test.reactormixins import ReactorBuilder
from twisted.trial.unittest import SkipTest, TestCase
@skipIf(noGtkSkip, noGtkMessage)
def test_gtkApplicationActivate(self) -> None:
    """
        L{Gtk.Application} instances can be registered with a gtk3reactor.
        """
    self.reactorFactory = gireactor.GIReactor
    reactor = self.buildReactor()
    app = Gtk.Application(application_id='com.twistedmatrix.trial.gtk3reactor', flags=Gio.ApplicationFlags.FLAGS_NONE)
    self.runReactor(app, reactor)