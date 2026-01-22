from typing import Dict, List, Tuple
from twisted.internet.testing import StringTransport
from twisted.protocols import postfix
from twisted.trial import unittest
def test_chat(self):
    """
        Test that I{get} and I{put} commands are responded to correctly by
        L{postfix.PostfixTCPMapServer} when its factory is an instance of
        L{postifx.PostfixTCPMapDictServerFactory}.
        """
    factory = postfix.PostfixTCPMapDictServerFactory(self.data)
    transport = StringTransport()
    protocol = postfix.PostfixTCPMapServer()
    protocol.service = factory
    protocol.factory = factory
    protocol.makeConnection(transport)
    for input, expected_output in self.chat:
        protocol.lineReceived(input)
        self.assertEqual(transport.value(), expected_output, 'For %r, expected %r but got %r' % (input, expected_output, transport.value()))
        transport.clear()
    protocol.setTimeout(None)