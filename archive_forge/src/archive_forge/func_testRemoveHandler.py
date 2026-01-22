import gc
import io
import os
import sys
import signal
import weakref
import unittest
from test import support
def testRemoveHandler(self):
    default_handler = signal.getsignal(signal.SIGINT)
    unittest.installHandler()
    unittest.removeHandler()
    self.assertEqual(signal.getsignal(signal.SIGINT), default_handler)
    unittest.removeHandler()
    self.assertEqual(signal.getsignal(signal.SIGINT), default_handler)