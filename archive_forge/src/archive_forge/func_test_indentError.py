import builtins
import struct
from io import StringIO
from twisted.internet import defer, error
from twisted.internet.testing import StringTransport
from twisted.protocols import ident
from twisted.python import failure
from twisted.trial import unittest
def test_indentError(self):
    """
        'UNKNOWN-ERROR' error should map to the L{ident.IdentError} exception.
        """
    d = defer.Deferred()
    self.client.queries.append((d, 123, 456))
    self.client.lineReceived('123, 456 : ERROR : UNKNOWN-ERROR')
    return self.assertFailure(d, ident.IdentError)