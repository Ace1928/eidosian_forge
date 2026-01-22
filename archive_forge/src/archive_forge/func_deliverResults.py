from socket import (
from typing import (
from zope.interface import implementer
from twisted.internet._idna import _idnaBytes
from twisted.internet.address import IPv4Address, IPv6Address
from twisted.internet.defer import Deferred
from twisted.internet.error import DNSLookupError
from twisted.internet.interfaces import (
from twisted.internet.threads import deferToThreadPool
from twisted.logger import Logger
from twisted.python.compat import nativeString
@d.addCallback
def deliverResults(result: _GETADDRINFO_RESULT) -> None:
    for family, socktype, proto, cannoname, sockaddr in result:
        addrType = _afToType[family]
        resolutionReceiver.addressResolved(addrType(_socktypeToType.get(socktype, 'TCP'), *sockaddr))
    resolutionReceiver.resolutionComplete()