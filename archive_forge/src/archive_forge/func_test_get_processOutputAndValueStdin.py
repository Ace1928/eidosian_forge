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
def test_get_processOutputAndValueStdin(self):
    """
        Standard input can be made available to the child process by passing
        bytes for the `stdinBytes` parameter.
        """
    scriptFile = self.makeSourceFile(['import sys', 'sys.stdout.write(sys.stdin.read())'])
    stdinBytes = b'These are the bytes to see.'
    d = utils.getProcessOutputAndValue(self.exe, ['-u', scriptFile], stdinBytes=stdinBytes)

    def gotOutputAndValue(out_err_code):
        out, err, code = out_err_code
        self.assertIn(stdinBytes, out)
        self.assertEqual(0, code)
    d.addCallback(gotOutputAndValue)
    return d