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
def test_fileDescriptorOverrun(self):
    """
        If L{IUNIXTransport.sendFileDescriptor} is used to queue a greater
        number of file descriptors than the number of bytes sent using
        L{ITransport.write}, the connection is closed and the protocol connected
        to the transport has its C{connectionLost} method called with a failure
        wrapping L{FileDescriptorOverrun}.
        """
    cargo = socket()
    server = SendFileDescriptor(cargo.fileno(), None)
    client = ReceiveFileDescriptor()
    result = []
    d = client.waitForDescriptor()
    d.addBoth(result.append)
    d.addBoth(lambda ignored: server.transport.loseConnection())
    runProtocolsWithReactor(self, server, client, self.endpoints)
    self.assertIsInstance(result[0], Failure)
    result[0].trap(ConnectionClosed)
    self.assertIsInstance(server.reason.value, FileDescriptorOverrun)