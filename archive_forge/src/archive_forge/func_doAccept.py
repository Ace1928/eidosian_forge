from __future__ import annotations
import errno
import socket
import struct
from typing import TYPE_CHECKING, Optional, Union
from zope.interface import classImplements, implementer
from twisted.internet import address, defer, error, interfaces, main
from twisted.internet.abstract import _LogOwner, isIPv6Address
from twisted.internet.address import IPv4Address, IPv6Address
from twisted.internet.interfaces import IProtocol
from twisted.internet.iocpreactor import abstract, iocpsupport as _iocp
from twisted.internet.iocpreactor.const import (
from twisted.internet.iocpreactor.interfaces import IReadWriteHandle
from twisted.internet.protocol import Protocol
from twisted.internet.tcp import (
from twisted.python import failure, log, reflect
def doAccept(self):
    evt = _iocp.Event(self.cbAccept, self)
    evt.buff = buff = bytearray(2 * (self.addrLen + 16))
    evt.newskt = newskt = self.reactor.createSocket(self.addressFamily, self.socketType)
    rc = _iocp.accept(self.socket.fileno(), newskt.fileno(), buff, evt)
    if rc and rc != ERROR_IO_PENDING:
        self.handleAccept(rc, evt)