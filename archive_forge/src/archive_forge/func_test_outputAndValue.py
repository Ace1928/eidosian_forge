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
def test_outputAndValue(self):
    """
        The L{Deferred} returned by L{getProcessOutputAndValue} fires with a
        three-tuple, the elements of which give the data written to the child's
        stdout, the data written to the child's stderr, and the exit status of
        the child.
        """
    scriptFile = self.makeSourceFile(['import sys', "sys.stdout.buffer.write(b'hello world!\\n')", "sys.stderr.buffer.write(b'goodbye world!\\n')", 'sys.exit(1)'])

    def gotOutputAndValue(out_err_code):
        out, err, code = out_err_code
        self.assertEqual(out, b'hello world!\n')
        self.assertEqual(err, b'goodbye world!\n')
        self.assertEqual(code, 1)
    d = utils.getProcessOutputAndValue(self.exe, ['-u', scriptFile])
    return d.addCallback(gotOutputAndValue)