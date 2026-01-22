from __future__ import annotations
from unittest import skipIf
from twisted.internet.error import ReactorAlreadyRunning
from twisted.internet.test.reactormixins import ReactorBuilder
from twisted.trial.unittest import SkipTest, TestCase
def test_noQuit(self) -> None:
    """
        Older versions of PyGObject lack C{Application.quit}, and so won't
        allow registration.
        """
    self.reactorFactory = lambda: gireactor.GIReactor(useGtk=False)
    reactor = self.buildReactor()
    app = object()
    exc = self.assertRaises(RuntimeError, reactor.registerGApplication, app)
    self.assertTrue(exc.args[0].startswith('Application registration is not'))