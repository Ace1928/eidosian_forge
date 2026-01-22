import builtins
import struct
from io import StringIO
from twisted.internet import defer, error
from twisted.internet.testing import StringTransport
from twisted.protocols import ident
from twisted.python import failure
from twisted.trial import unittest
def testDottedQuadFromHexString(self):
    p = ident.ProcServerMixin()
    self.assertEqual(p.dottedQuadFromHexString(_addr1), '127.0.0.1')