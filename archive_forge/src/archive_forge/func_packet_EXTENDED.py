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
def packet_EXTENDED(self, data):
    requestId = data[:4]
    data = data[4:]
    extName, extData = getNS(data)
    d = defer.maybeDeferred(self.client.extendedRequest, extName, extData)
    d.addCallback(self._cbExtended, requestId)
    d.addErrback(self._ebStatus, requestId, b'extended ' + extName + b' failed')