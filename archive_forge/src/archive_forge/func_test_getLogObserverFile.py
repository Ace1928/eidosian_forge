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
def test_getLogObserverFile(self):
    """
        When C{logfile} contains a file name, L{app.AppLogger._getLogObserver}
        returns a log observer pointing at the specified path, and a signal
        handler rotating the log is installed.
        """
    logFiles = _patchTextFileLogObserver(self.patch)
    filename = self.mktemp()
    sut = UnixAppLogger({'logfile': filename})
    observer = sut._getLogObserver()
    self.addCleanup(observer._outFile.close)
    self.assertEqual(len(logFiles), 1)
    self.assertEqual(logFiles[0].path, os.path.abspath(filename))
    self.assertEqual(len(self.signals), 1)
    self.assertEqual(self.signals[0][0], signal.SIGUSR1)
    d = Deferred()

    def rotate():
        d.callback(None)
    logFiles[0].rotate = rotate
    rotateLog = self.signals[0][1]
    rotateLog(None, None)
    return d