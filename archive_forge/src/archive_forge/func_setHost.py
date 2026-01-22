from __future__ import annotations
from io import BytesIO
from typing import Dict, List, Optional
from zope.interface import implementer, verify
from incremental import Version
from twisted.internet.address import IPv4Address, IPv6Address
from twisted.internet.defer import Deferred
from twisted.internet.interfaces import IAddress, ISSLTransport
from twisted.internet.task import Clock
from twisted.python.deprecate import deprecated
from twisted.trial import unittest
from twisted.web._responses import FOUND
from twisted.web.http_headers import Headers
from twisted.web.resource import Resource
from twisted.web.server import NOT_DONE_YET, Session, Site
def setHost(self, host, port, ssl=0):
    """
        Change the host and port the request thinks it's using.

        @type host: C{bytes}
        @param host: The value to which to change the host header.

        @type ssl: C{bool}
        @param ssl: A flag which, if C{True}, indicates that the request is
            considered secure (if C{True}, L{isSecure} will return C{True}).
        """
    self._forceSSL = ssl
    if self.isSecure():
        default = 443
    else:
        default = 80
    if port == default:
        hostHeader = host
    else:
        hostHeader = b'%b:%d' % (host, port)
    self.requestHeaders.addRawHeader(b'host', hostHeader)