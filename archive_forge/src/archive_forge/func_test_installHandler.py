from __future__ import annotations
import errno
import os
import signal
from twisted.python.runtime import platformType
from twisted.trial.unittest import SynchronousTestCase
def test_installHandler(self) -> None:
    """
        The file descriptor passed to L{installHandler} has a byte written to
        it when SIGCHLD is delivered to the process.
        """
    read, write = self.pipe()
    installHandler(write)
    exc = self.assertRaises(OSError, os.read, read, 1)
    self.assertEqual(exc.errno, errno.EAGAIN)
    os.kill(os.getpid(), signal.SIGCHLD)
    self.assertEqual(len(os.read(read, 5)), 1)