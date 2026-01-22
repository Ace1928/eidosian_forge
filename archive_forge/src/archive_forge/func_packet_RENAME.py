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
def packet_RENAME(self, data):
    requestId = data[:4]
    data = data[4:]
    oldPath, data = getNS(data)
    newPath, data = getNS(data)
    assert data == b'', f'still have data in RENAME: {data!r}'
    d = defer.maybeDeferred(self.client.renameFile, oldPath, newPath)
    d.addCallback(self._cbStatus, requestId, b'rename succeeded')
    d.addErrback(self._ebStatus, requestId, b'rename failed')