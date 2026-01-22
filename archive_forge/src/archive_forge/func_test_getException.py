from typing import Dict, List, Tuple
from twisted.internet.testing import StringTransport
from twisted.protocols import postfix
from twisted.trial import unittest
def test_getException(self):
    """
        If the factory throws an exception,
        error code 400 must be returned.
        """

    class ErrorFactory:
        """
            Factory that raises an error on key lookup.
            """

        def get(self, key):
            raise Exception('This is a test error')
    server = postfix.PostfixTCPMapServer()
    server.factory = ErrorFactory()
    server.transport = StringTransport()
    server.lineReceived(b'get example')
    self.assertEqual(server.transport.value(), b'400 This is a test error\n')