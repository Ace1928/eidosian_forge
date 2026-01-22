from zope.interface.verify import verifyClass
from twisted.internet import defer
from twisted.internet.interfaces import IProtocolFactory
from twisted.names import dns, error, resolve, server
from twisted.python import failure, log
from twisted.trial import unittest
def test_sendReplyLoggingNoAnswers(self):
    """
        If L{server.DNSServerFactory.sendReply} logs a "no answers" message if
        the supplied message has no answers.
        """
    self.patch(server.time, 'time', lambda: 86402)
    m = dns.Message()
    m.timeReceived = 86401
    f = server.DNSServerFactory(verbose=2)
    assertLogMessage(self, ['Replying with no answers', 'Processed query in 1.000 seconds'], f.sendReply, protocol=NoopProtocol(), message=m, address=None)