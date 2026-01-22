import errno
import os
import sys
from twisted.python.util import untilConcludes
from twisted.trial import unittest
import os, errno
def test_setNonBlocking(self):
    """
        L{fdesc.setNonBlocking} sets a file description to non-blocking.
        """
    r, w = os.pipe()
    self.addCleanup(os.close, r)
    self.addCleanup(os.close, w)
    self.assertFalse(fcntl.fcntl(r, fcntl.F_GETFL) & os.O_NONBLOCK)
    fdesc.setNonBlocking(r)
    self.assertTrue(fcntl.fcntl(r, fcntl.F_GETFL) & os.O_NONBLOCK)