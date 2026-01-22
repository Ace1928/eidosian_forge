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
def writeChunk(self, offset, chunk):
    data = self.handle + struct.pack('!Q', offset) + NS(chunk)
    return self.parent._sendRequest(FXP_WRITE, data)