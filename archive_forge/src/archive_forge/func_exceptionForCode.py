import socket
from zope.interface import implementer
from twisted.internet import defer, error, interfaces
from twisted.logger import Logger
from twisted.names import dns
from twisted.names.error import (
def exceptionForCode(self, responseCode):
    """
        Convert a response code (one of the possible values of
        L{dns.Message.rCode} to an exception instance representing it.

        @since: 10.0
        """
    return self._errormap.get(responseCode, DNSUnknownError)