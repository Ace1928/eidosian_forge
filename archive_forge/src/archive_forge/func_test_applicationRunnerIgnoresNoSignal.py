import errno
import inspect
import os
import pickle
import signal
import sys
from io import StringIO
from unittest import skipIf
from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted import internet, logger, plugin
from twisted.application import app, reactors, service
from twisted.application.service import IServiceMaker
from twisted.internet.base import ReactorBase
from twisted.internet.defer import Deferred
from twisted.internet.interfaces import IReactorDaemonize, _ISupportsExitSignalCapturing
from twisted.internet.test.modulehelpers import AlternateReactor
from twisted.internet.testing import MemoryReactor
from twisted.logger import ILogObserver, globalLogBeginner, globalLogPublisher
from twisted.python import util
from twisted.python.components import Componentized
from twisted.python.fakepwd import UserDatabase
from twisted.python.log import ILogObserver as LegacyILogObserver, textFromEventDict
from twisted.python.reflect import requireModule
from twisted.python.runtime import platformType
from twisted.python.usage import UsageError
from twisted.scripts import twistd
from twisted.test.test_process import MockOS
from twisted.trial.unittest import TestCase
def test_applicationRunnerIgnoresNoSignal(self):
    """
        The runner sets its _exitSignal instance attribute to None if
        the reactor does not implement L{_ISupportsExitSignalCapturing}.
        """

    class DummyReactorWithExitSignalAttribute:
        """
            A dummy reactor, providing a C{run} method, and setting the
            _exitSignal attribute to a nonzero value.
            """

        def installWaker(self):
            """
                Dummy method, does nothing.
                """

        def run(self):
            """
                A fake run method setting _exitSignal to a nonzero value
                that should be ignored.
                """
            self._exitSignal = 2
    reactor = DummyReactorWithExitSignalAttribute()
    runner = app.ApplicationRunner({'profile': False, 'profiler': 'profile', 'debug': False})
    runner.startReactor(reactor, None, None)
    self.assertEquals(None, runner._exitSignal)