import errno
import gc
import gzip
import operator
import os
import signal
import stat
import sys
from unittest import SkipTest, skipIf
from io import BytesIO
from zope.interface.verify import verifyObject
from twisted.internet import defer, error, interfaces, protocol, reactor
from twisted.python import procutils, runtime
from twisted.python.compat import networkString
from twisted.python.filepath import FilePath
from twisted.python.log import msg
from twisted.trial import unittest
def test_inConnectionLost(self):
    """
        Verify that when stdin close notification is delivered to
        L{ProcessProtocol.childConnectionLost}, it is forwarded to
        L{ProcessProtocol.inConnectionLost}.
        """
    lost = []

    class InLostProtocol(StubProcessProtocol):

        def inConnectionLost(self):
            lost.append(None)
    p = InLostProtocol()
    p.childConnectionLost(0)
    self.assertEqual(lost, [None])