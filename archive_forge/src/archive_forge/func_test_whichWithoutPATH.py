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
def test_whichWithoutPATH(self):
    """
        Test that if C{os.environ} does not have a C{'PATH'} key,
        L{procutils.which} returns an empty list.
        """
    del os.environ['PATH']
    self.assertEqual(procutils.which('executable'), [])