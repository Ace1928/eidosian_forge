from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted.conch import telnet
from twisted.internet import defer
from twisted.python.compat import iterbytes
from twisted.test import proto_helpers
from twisted.trial import unittest
def test_requestNegotiationEscapesIAC(self):
    """
        If the payload for a subnegotiation includes I{IAC}, it is escaped by
        L{telnet.Telnet.requestNegotiation} with another I{IAC}.

        See RFC 855.
        """
    transport = proto_helpers.StringTransport()
    self.protocol.makeConnection(transport)
    self.protocol.requestNegotiation(b'\x01', b'\xff')
    self.assertEqual(transport.value(), b'\xff\xfa\x01\xff\xff\xff\xf0')