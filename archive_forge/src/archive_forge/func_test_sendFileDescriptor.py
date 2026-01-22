from hashlib import md5
from os import close, fstat, stat, unlink, urandom
from pprint import pformat
from socket import AF_INET, SOCK_STREAM, SOL_SOCKET, socket
from stat import S_IMODE
from struct import pack
from tempfile import mkstemp, mktemp
from typing import Optional, Sequence, Type
from unittest import skipIf
from zope.interface import Interface, implementer
from twisted.internet import base, interfaces
from twisted.internet.address import UNIXAddress
from twisted.internet.defer import Deferred, fail, gatherResults
from twisted.internet.endpoints import UNIXClientEndpoint, UNIXServerEndpoint
from twisted.internet.error import (
from twisted.internet.interfaces import (
from twisted.internet.protocol import ClientFactory, DatagramProtocol, ServerFactory
from twisted.internet.task import LoopingCall
from twisted.internet.test.connectionmixins import (
from twisted.internet.test.reactormixins import ReactorBuilder
from twisted.internet.test.test_tcp import (
from twisted.python.compat import nativeString
from twisted.python.failure import Failure
from twisted.python.filepath import _coerceToFilesystemEncoding
from twisted.python.log import addObserver, err, removeObserver
from twisted.python.reflect import requireModule
from twisted.python.runtime import platform
@skipIf(not sendmsg, sendmsgSkipReason)
def test_sendFileDescriptor(self):
    """
        L{IUNIXTransport.sendFileDescriptor} accepts an integer file descriptor
        and sends a copy of it to the process reading from the connection.
        """
    from socket import fromfd
    s = socket()
    s.bind(('', 0))
    server = SendFileDescriptor(s.fileno(), b'junk')
    client = ReceiveFileDescriptor()
    d = client.waitForDescriptor()

    def checkDescriptor(descriptor):
        received = fromfd(descriptor, AF_INET, SOCK_STREAM)
        close(descriptor)
        self.assertEqual(s.getsockname(), received.getsockname())
        self.assertNotEqual(s.fileno(), received.fileno())
    d.addCallback(checkDescriptor)
    d.addErrback(err, 'Sending file descriptor encountered a problem')
    d.addBoth(lambda ignored: server.transport.loseConnection())
    runProtocolsWithReactor(self, server, client, self.endpoints)