import builtins
import struct
from io import StringIO
from twisted.internet import defer, error
from twisted.internet.testing import StringTransport
from twisted.protocols import ident
from twisted.python import failure
from twisted.trial import unittest
def testNonExistingAddress(self):
    p = ident.ProcServerMixin()
    p.entries = lambda: iter([self.line])
    self.assertRaises(ident.NoUser, p.lookup, ('127.0.0.1', 26), ('1.2.3.4', 762))
    self.assertRaises(ident.NoUser, p.lookup, ('127.0.0.1', 25), ('1.2.3.5', 762))
    self.assertRaises(ident.NoUser, p.lookup, ('127.0.0.1', 25), ('1.2.3.4', 763))