import errno
import os
import sys
import warnings
from os import close, pathsep, pipe, read
from socket import AF_INET, AF_INET6, SOL_SOCKET, error, socket
from struct import pack
from unittest import skipIf
from twisted.internet import reactor
from twisted.internet.defer import Deferred, inlineCallbacks
from twisted.internet.error import ProcessDone
from twisted.internet.protocol import ProcessProtocol
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.trial.unittest import TestCase
@inlineCallbacks
def test_sendSubProcessFD(self):
    """
        Calling L{sendmsg} with SOL_SOCKET, SCM_RIGHTS, and a platform-endian
        packed file descriptor number should send that file descriptor to a
        different process, where it can be retrieved by using L{recv1msg}.
        """
    sspp = _spawn('pullpipe', self.output.fileno())
    yield sspp.started
    pipeOut, pipeIn = _makePipe()
    self.addCleanup(pipeOut.close)
    self.addCleanup(pipeIn.close)
    with pipeIn:
        sendmsg(self.input, b'blonk', [(SOL_SOCKET, SCM_RIGHTS, pack('i', pipeIn.fileno()))])
    yield sspp.stopped
    self.assertEqual(read(pipeOut.fileno(), 1024), b'Test fixture data: blonk.\n')
    self.assertEqual(read(pipeOut.fileno(), 1024), b'')