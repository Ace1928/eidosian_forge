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
def test_FD(self):
    scriptPath = b'twisted.test.process_fds'
    d = defer.Deferred()
    p = FDChecker(d)
    reactor.spawnProcess(p, pyExe, [pyExe, b'-u', b'-m', scriptPath], env=properEnv, childFDs={0: 'w', 1: 'r', 2: 2, 3: 'w', 4: 'r', 5: 'w'})
    d.addCallback(lambda x: self.assertFalse(p.failed, p.failed))
    return d