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
def test_legacyObservers(self):
    """
        L{app.AppLogger} using a legacy logger observer still works, wrapping
        it in a compat shim.
        """
    logs = []
    logger = app.AppLogger({})

    @implementer(LegacyILogObserver)
    class LoggerObserver:
        """
            An observer which implements the legacy L{LegacyILogObserver}.
            """

        def __call__(self, x):
            """
                Add C{x} to the logs list.
                """
            logs.append(x)
    logger._observerFactory = lambda: LoggerObserver()
    logger.start(Componentized())
    self.assertIn('starting up', textFromEventDict(logs[0]))
    warnings = self.flushWarnings([self.test_legacyObservers])
    self.assertEqual(len(warnings), 0, warnings)