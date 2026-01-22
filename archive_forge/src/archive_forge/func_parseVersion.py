import re
from zope.interface import implementer
from twisted.internet.defer import (
from twisted.internet.error import ConnectionDone
from twisted.internet.interfaces import IConsumer, IPushProducer
from twisted.internet.protocol import Protocol
from twisted.logger import Logger
from twisted.protocols.basic import LineReceiver
from twisted.python.compat import networkString
from twisted.python.components import proxyForInterface
from twisted.python.failure import Failure
from twisted.python.reflect import fullyQualifiedName
from twisted.web.http import (
from twisted.web.http_headers import Headers
from twisted.web.iweb import UNKNOWN_LENGTH, IClientRequest, IResponse
def parseVersion(self, strversion):
    """
        Parse version strings of the form Protocol '/' Major '.' Minor. E.g.
        b'HTTP/1.1'.  Returns (protocol, major, minor).  Will raise ValueError
        on bad syntax.
        """
    try:
        proto, strnumber = strversion.split(b'/')
        major, minor = strnumber.split(b'.')
        major, minor = (int(major), int(minor))
    except ValueError as e:
        raise BadResponseVersion(str(e), strversion)
    if major < 0 or minor < 0:
        raise BadResponseVersion('version may not be negative', strversion)
    return (proto, major, minor)