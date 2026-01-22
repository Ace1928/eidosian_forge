import gc
import io
import os
import sys
import signal
import weakref
import unittest
from test import support
def testInterruptCaught(self):

    def test(result):
        pid = os.getpid()
        os.kill(pid, signal.SIGINT)
        result.breakCaught = True
        self.assertTrue(result.shouldStop)

    def test_function():
        result = unittest.TestResult()
        unittest.installHandler()
        unittest.registerResult(result)
        self.assertNotEqual(signal.getsignal(signal.SIGINT), self._default_handler)
        try:
            test(result)
        except KeyboardInterrupt:
            self.fail('KeyboardInterrupt not handled')
        self.assertTrue(result.breakCaught)
    self.withRepeats(test_function)