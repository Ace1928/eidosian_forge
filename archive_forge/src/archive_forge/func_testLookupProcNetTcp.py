import builtins
import struct
from io import StringIO
from twisted.internet import defer, error
from twisted.internet.testing import StringTransport
from twisted.protocols import ident
from twisted.python import failure
from twisted.trial import unittest
def testLookupProcNetTcp(self):
    """
        L{ident.ProcServerMixin.lookup} uses the Linux TCP process table.
        """
    open_calls = []

    def mocked_open(*args, **kwargs):
        """
            Mock for the open call to prevent actually opening /proc/net/tcp.
            """
        open_calls.append((args, kwargs))
        return StringIO(self.sampleFile)
    self.patch(builtins, 'open', mocked_open)
    p = ident.ProcServerMixin()
    self.assertRaises(ident.NoUser, p.lookup, ('127.0.0.1', 26), ('1.2.3.4', 762))
    self.assertEqual([(('/proc/net/tcp',), {})], open_calls)