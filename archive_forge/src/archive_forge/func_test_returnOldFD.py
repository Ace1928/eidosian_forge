from __future__ import annotations
import errno
import os
import signal
from twisted.python.runtime import platformType
from twisted.trial.unittest import SynchronousTestCase
def test_returnOldFD(self) -> None:
    """
        L{installHandler} returns the previously registered file descriptor.
        """
    read, write = self.pipe()
    oldFD = installHandler(write)
    self.assertEqual(installHandler(oldFD), write)