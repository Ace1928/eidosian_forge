import os
import signal
import stat
import sys
import warnings
from unittest import skipIf
from twisted.internet import error, interfaces, reactor, utils
from twisted.internet.defer import Deferred
from twisted.python.runtime import platform
from twisted.python.test.test_util import SuppressedWarningsTests
from twisted.trial.unittest import SynchronousTestCase, TestCase
def test_outputWithErrorCollected(self):
    """
        If a C{True} value is supplied for the C{errortoo} parameter to
        L{getProcessOutput}, the returned L{Deferred} fires with the child's
        stderr output as well as its stdout output.
        """
    scriptFile = self.makeSourceFile(['import sys', 'sys.stdout.write("foo")', 'sys.stdout.flush()', 'sys.stderr.write("foo")', 'sys.stderr.flush()'])
    d = utils.getProcessOutput(self.exe, ['-u', scriptFile], errortoo=True)
    return d.addCallback(self.assertEqual, b'foofoo')