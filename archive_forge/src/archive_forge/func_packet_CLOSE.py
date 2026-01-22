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
def packet_CLOSE(self, data):
    requestId = data[:4]
    data = data[4:]
    handle, data = getNS(data)
    self._log.info('closing: {requestId!r} {handle!r}', requestId=requestId, handle=handle)
    assert data == b'', f'still have data in CLOSE: {data!r}'
    if handle in self.openFiles:
        fileObj = self.openFiles[handle]
        d = defer.maybeDeferred(fileObj.close)
        d.addCallback(self._cbClose, handle, requestId)
        d.addErrback(self._ebStatus, requestId, b'close failed')
    elif handle in self.openDirs:
        dirObj = self.openDirs[handle][0]
        d = defer.maybeDeferred(dirObj.close)
        d.addCallback(self._cbClose, handle, requestId, 1)
        d.addErrback(self._ebStatus, requestId, b'close failed')
    else:
        code = errno.ENOENT
        text = os.strerror(code)
        err = OSError(code, text)
        self._ebStatus(failure.Failure(err), requestId)