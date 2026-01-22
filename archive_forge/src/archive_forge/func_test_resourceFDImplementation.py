import errno
import os
import sys
from typing import Optional
from twisted.trial.unittest import TestCase
def test_resourceFDImplementation(self):
    """
        L{_FDDetector._fallbackFDImplementation} uses the L{resource} module if
        it is available, returning a range of integers from 0 to the
        minimum of C{1024} and the hard I{NOFILE} limit.
        """
    self.revealResourceModule(512)
    self.assertEqual(list(range(512)), list(self.detector._fallbackFDImplementation()))
    self.revealResourceModule(2048)
    self.assertEqual(list(range(1024)), list(self.detector._fallbackFDImplementation()))