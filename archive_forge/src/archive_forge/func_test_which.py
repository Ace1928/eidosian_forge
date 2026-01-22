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
def test_which(self):
    j = os.path.join
    paths = procutils.which('executable')
    expectedPaths = [j(self.foobaz, 'executable'), j(self.bazfoo, 'executable')]
    if runtime.platform.isWindows():
        expectedPaths.append(j(self.bazbar, 'executable'))
    self.assertEqual(paths, expectedPaths)