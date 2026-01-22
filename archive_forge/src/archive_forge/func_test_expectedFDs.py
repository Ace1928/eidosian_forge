import errno
import os
import sys
from typing import Optional
from twisted.trial.unittest import TestCase
def test_expectedFDs(self):
    """
        L{_listOpenFDs} lists expected file descriptors.
        """
    f = open(os.devnull)
    openfds = process._listOpenFDs()
    self.assertIn(f.fileno(), openfds)
    fd = os.dup(f.fileno())
    self.assertTrue(fd > f.fileno(), 'Expected duplicate file descriptor to be greater than original')
    try:
        f.close()
        self.assertIn(fd, process._listOpenFDs())
    finally:
        os.close(fd)
    self.assertNotIn(fd, process._listOpenFDs())