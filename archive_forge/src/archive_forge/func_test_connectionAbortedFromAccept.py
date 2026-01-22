import errno
import os
import socket
from unittest import skipIf
from twisted.internet import interfaces, reactor
from twisted.internet.defer import gatherResults, maybeDeferred
from twisted.internet.protocol import Protocol, ServerFactory
from twisted.internet.tcp import (
from twisted.python import log
from twisted.python.runtime import platform
from twisted.trial.unittest import TestCase
def test_connectionAbortedFromAccept(self):
    """
        Similar to L{test_tooManyFilesFromAccept}, but test the case where
        C{accept(2)} fails with C{ECONNABORTED}.

        It is not clear whether this is actually possible for TCP
        connections on modern versions of Linux.
        """
    return self._acceptFailureTest(ECONNABORTED)