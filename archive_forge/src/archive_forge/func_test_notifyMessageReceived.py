from zope.interface.verify import verifyClass
from twisted.internet import defer
from twisted.internet.interfaces import IProtocolFactory
from twisted.names import dns, error, resolve, server
from twisted.python import failure, log
from twisted.trial import unittest
def test_notifyMessageReceived(self):
    """
        L{DNSServerFactory.messageReceived} passes messages with an opcode of
        C{OP_NOTIFY} on to L{DNSServerFactory.handleNotify}.
        """
    self._messageReceivedTest('handleNotify', dns.Message(opCode=dns.OP_NOTIFY))