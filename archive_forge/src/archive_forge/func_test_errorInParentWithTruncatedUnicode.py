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
def test_errorInParentWithTruncatedUnicode(self):
    """
        When the child writes a non-ASCII error message to the status
        pipe during daemonization, and that message is too longer, the
        parent writes the repr of the truncated message to C{stderr}
        and exits with a non-zero status code.
        """
    truncatedMessage = b'1 RuntimeError: ' + b'\\u2022' * 14
    reportedMessage = "b'RuntimeError: {}'".format('\\\\u2022' * 14)
    self.assertErrorInParentBehavior(readData=truncatedMessage, errorMessage='An error has occurred: {}\nPlease look at log file for more information.\n'.format(reportedMessage), mockOSActions=[('chdir', '.'), ('umask', 63), ('fork', True), ('read', -1, 100), ('exit', 1), ('unlink', 'twistd.pid')])