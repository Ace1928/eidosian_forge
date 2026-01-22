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
@skipIf(sys.stdout.encoding != sys.getfilesystemencoding(), 'sys.stdout.encoding: {} does not match sys.getfilesystemencoding(): {} .  May need to set PYTHONUTF8 and PYTHONIOENCODING environment variables.'.format(sys.stdout.encoding, sys.getfilesystemencoding()))
def test_UTF8StringInEnvironment(self):
    """
        L{os.environ} (inherited by every subprocess on Windows) can
        contain a UTF-8 string value.
        """
    envKey = 'TWISTED_BUILD_SOURCEVERSIONAUTHOR'
    envKeyBytes = b'TWISTED_BUILD_SOURCEVERSIONAUTHOR'
    envVal = 'Speciał Committór'
    os.environ[envKey] = envVal
    self.addCleanup(operator.delitem, os.environ, envKey)
    p = GetEnvironmentDictionary.run(reactor, [], os.environ)

    def gotEnvironment(environb):
        self.assertIn(envKeyBytes, environb)
        self.assertEqual(environb[envKeyBytes], 'Speciał Committór'.encode(sys.stdout.encoding))
    return p.getResult().addCallback(gotEnvironment)