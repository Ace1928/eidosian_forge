import errno
import socket
from zope.interface import verify
from twisted.internet.error import UnsupportedAddressFamily
from twisted.internet.interfaces import IReactorSocket
from twisted.internet.protocol import DatagramProtocol, ServerFactory
from twisted.internet.test.reactormixins import ReactorBuilder, needsRunningReactor
from twisted.python.log import err
from twisted.python.runtime import platform

        When the L{IListeningPort} returned by
        L{IReactorSocket.adoptDatagramPort} is stopped using
        C{stopListening}, the underlying socket is closed but not
        shutdown.  This allows another process which still has a
        reference to it to continue reading and writing to it.
        