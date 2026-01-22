import builtins
import struct
from io import StringIO
from twisted.internet import defer, error
from twisted.internet.testing import StringTransport
from twisted.protocols import ident
from twisted.python import failure
from twisted.trial import unittest
def test_noUSerError(self):
    """
        'NO-USER' error should map to the L{ident.NoUser} exception.
        """
    d = defer.Deferred()
    self.client.queries.append((d, 234, 456))
    self.client.lineReceived('234, 456 : ERROR : NO-USER')
    return self.assertFailure(d, ident.NoUser)