import os
import socket
from unittest import skipIf
from twisted.internet.address import (
from twisted.python.compat import nativeString
from twisted.python.runtime import platform
from twisted.trial.unittest import SynchronousTestCase, TestCase
@skipIf(symlinkSkip, 'Platform does not support symlinks')
def test_hashOfLinkedFiles(self):
    """
        UNIXAddress Objects that compare as equal have the same hash value.
        """
    linkName = self.mktemp()
    with open(self._socketAddress, 'w') as self.fd:
        os.symlink(os.path.abspath(self._socketAddress), linkName)
        self.assertEqual(hash(UNIXAddress(self._socketAddress)), hash(UNIXAddress(linkName)))