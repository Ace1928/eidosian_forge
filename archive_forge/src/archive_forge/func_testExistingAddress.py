import builtins
import struct
from io import StringIO
from twisted.internet import defer, error
from twisted.internet.testing import StringTransport
from twisted.protocols import ident
from twisted.python import failure
from twisted.trial import unittest
def testExistingAddress(self):
    username = []
    p = ident.ProcServerMixin()
    p.entries = lambda: iter([self.line])
    p.getUsername = lambda uid: (username.append(uid), 'root')[1]
    self.assertEqual(p.lookup(('127.0.0.1', 25), ('1.2.3.4', 762)), (p.SYSTEM_NAME, 'root'))
    self.assertEqual(username, [0])