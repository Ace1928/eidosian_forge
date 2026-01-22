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
def test_withoutProfile(self):
    """
        When the C{profile} module is not present, L{app.ProfilerRunner.run}
        should raise a C{SystemExit} exception.
        """
    savedModules = sys.modules.copy()
    config = twistd.ServerOptions()
    config['profiler'] = 'profile'
    profiler = app.AppProfiler(config)
    sys.modules['profile'] = None
    try:
        self.assertRaises(SystemExit, profiler.run, None)
    finally:
        sys.modules.clear()
        sys.modules.update(savedModules)