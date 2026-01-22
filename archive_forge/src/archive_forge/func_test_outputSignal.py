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
@skipIf(platform.isWindows(), "Windows doesn't have real signals.")
def test_outputSignal(self):
    """
        If the child process exits because of a signal, the L{Deferred}
        returned by L{getProcessOutputAndValue} fires a L{Failure} of a tuple
        containing the child's stdout, stderr, and the signal which caused
        it to exit.
        """
    scriptFile = self.makeSourceFile(['import sys, os, signal', "sys.stdout.write('stdout bytes\\n')", "sys.stderr.write('stderr bytes\\n')", 'sys.stdout.flush()', 'sys.stderr.flush()', 'os.kill(os.getpid(), signal.SIGKILL)'])

    def gotOutputAndValue(out_err_sig):
        out, err, sig = out_err_sig
        self.assertEqual(out, b'stdout bytes\n')
        self.assertEqual(err, b'stderr bytes\n')
        self.assertEqual(sig, signal.SIGKILL)
    d = utils.getProcessOutputAndValue(self.exe, ['-u', scriptFile])
    d = self.assertFailure(d, tuple)
    return d.addCallback(gotOutputAndValue)