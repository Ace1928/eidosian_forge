import gc
import os
import sys
from twisted.trial._dist.options import WorkerOptions
from twisted.trial.unittest import TestCase
def test_coverage(self) -> None:
    """
        L{WorkerOptions.coverdir} returns the C{coverage} child directory of
        the current directory to be used for storing coverage data.
        """
    self.assertEqual(os.path.realpath(os.path.join(os.getcwd(), 'coverage')), self.options.coverdir().path)