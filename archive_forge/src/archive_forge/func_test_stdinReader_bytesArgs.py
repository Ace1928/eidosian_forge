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
def test_stdinReader_bytesArgs(self):
    """
        Pass L{bytes} args to L{_test_stdinReader}.
        """
    import win32api
    pyExe = FilePath(sys.executable)._asBytesPath()
    args = [pyExe, b'-u', b'-m', b'twisted.test.process_stdinreader']
    env = dict(os.environ)
    env[b'PYTHONPATH'] = os.pathsep.join(sys.path).encode(sys.getfilesystemencoding())
    path = win32api.GetTempPath()
    path = path.encode(sys.getfilesystemencoding())
    d = self._test_stdinReader(pyExe, args, env, path)
    return d