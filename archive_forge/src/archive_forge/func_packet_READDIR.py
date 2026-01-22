import errno
import os
import struct
import warnings
from typing import Dict
from zope.interface import implementer
from twisted.conch.interfaces import ISFTPFile, ISFTPServer
from twisted.conch.ssh.common import NS, getNS
from twisted.internet import defer, error, protocol
from twisted.logger import Logger
from twisted.python import failure
from twisted.python.compat import nativeString, networkString
def packet_READDIR(self, data):
    requestId = data[:4]
    data = data[4:]
    handle, data = getNS(data)
    assert data == b'', f'still have data in READDIR: {data!r}'
    if handle not in self.openDirs:
        self._ebStatus(failure.Failure(KeyError()), requestId)
    else:
        dirObj, dirIter = self.openDirs[handle]
        d = defer.maybeDeferred(self._scanDirectory, dirIter, [])
        d.addCallback(self._cbSendDirectory, requestId)
        d.addErrback(self._ebStatus, requestId, b'scan directory failed')