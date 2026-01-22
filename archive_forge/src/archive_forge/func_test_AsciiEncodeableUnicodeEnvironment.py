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
def test_AsciiEncodeableUnicodeEnvironment(self):
    """
        C{os.environ} (inherited by every subprocess on Windows)
        contains Unicode keys and Unicode values which can be ASCII-encodable.
        """
    os.environ['KEY_ASCII'] = 'VALUE_ASCII'
    self.addCleanup(operator.delitem, os.environ, 'KEY_ASCII')
    p = GetEnvironmentDictionary.run(reactor, [], os.environ)

    def gotEnvironment(environb):
        self.assertEqual(environb[b'KEY_ASCII'], b'VALUE_ASCII')
    return p.getResult().addCallback(gotEnvironment)