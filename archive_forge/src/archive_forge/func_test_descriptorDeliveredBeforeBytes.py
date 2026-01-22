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
def test_descriptorDeliveredBeforeBytes(self):
    """
        L{IUNIXTransport.sendFileDescriptor} sends file descriptors before
        L{ITransport.write} sends normal bytes.
        """

    @implementer(IFileDescriptorReceiver)
    class RecordEvents(ConnectableProtocol):

        def connectionMade(self):
            ConnectableProtocol.connectionMade(self)
            self.events = []

        def fileDescriptorReceived(innerSelf, descriptor):
            self.addCleanup(close, descriptor)
            innerSelf.events.append(type(descriptor))

        def dataReceived(self, data):
            self.events.extend(data)
    cargo = socket()
    server = SendFileDescriptor(cargo.fileno(), b'junk')
    client = RecordEvents()
    runProtocolsWithReactor(self, server, client, self.endpoints)
    self.assertEqual(int, client.events[0])
    self.assertEqual(b'junk', bytes(client.events[1:]))