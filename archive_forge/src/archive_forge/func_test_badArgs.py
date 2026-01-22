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
def test_badArgs(self):
    pyArgs = [pyExe, b'-u', b'-c', b"print('hello')"]
    p = Accumulator()
    self.assertRaises(ValueError, reactor.spawnProcess, p, pyExe, pyArgs, uid=1)
    self.assertRaises(ValueError, reactor.spawnProcess, p, pyExe, pyArgs, gid=1)
    self.assertRaises(ValueError, reactor.spawnProcess, p, pyExe, pyArgs, usePTY=1)
    self.assertRaises(ValueError, reactor.spawnProcess, p, pyExe, pyArgs, childFDs={1: 'r'})