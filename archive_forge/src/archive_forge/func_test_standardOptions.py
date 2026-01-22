import gc
import os
import sys
from twisted.trial._dist.options import WorkerOptions
from twisted.trial.unittest import TestCase
def test_standardOptions(self) -> None:
    """
        L{WorkerOptions} supports a subset of standard options supported by
        trial.
        """
    self.addCleanup(sys.setrecursionlimit, sys.getrecursionlimit())
    if gc.isenabled():
        self.addCleanup(gc.enable)
    gc.enable()
    self.options.parseOptions(['--recursionlimit', '2000', '--disablegc'])
    self.assertEqual(2000, sys.getrecursionlimit())
    self.assertFalse(gc.isenabled())