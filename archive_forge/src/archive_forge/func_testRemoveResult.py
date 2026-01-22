import gc
import io
import os
import sys
import signal
import weakref
import unittest
from test import support
def testRemoveResult(self):
    result = unittest.TestResult()
    unittest.registerResult(result)
    unittest.installHandler()
    self.assertTrue(unittest.removeResult(result))
    self.assertFalse(unittest.removeResult(unittest.TestResult()))
    try:
        pid = os.getpid()
        os.kill(pid, signal.SIGINT)
    except KeyboardInterrupt:
        pass
    self.assertFalse(result.shouldStop)