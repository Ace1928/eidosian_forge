import fcntl
import fnmatch
import getpass
import glob
import os
import pwd
import stat
import struct
import sys
import tty
from typing import List, Optional, TextIO, Union
from twisted.conch.client import connect, default, options
from twisted.conch.ssh import channel, common, connection, filetransfer
from twisted.internet import defer, reactor, stdio, utils
from twisted.protocols import basic
from twisted.python import failure, log, usage
from twisted.python.filepath import FilePath
def cmd_GET(self, rest):
    remote, rest = self._getFilename(rest)
    if '*' in remote or '?' in remote:
        if rest:
            local, rest = self._getFilename(rest)
            if not os.path.isdir(local):
                return 'Wildcard get with non-directory target.'
        else:
            local = b''
        d = self._remoteGlob(remote)
        d.addCallback(self._cbGetMultiple, local)
        return d
    if rest:
        local, rest = self._getFilename(rest)
    else:
        local = os.path.split(remote)[1]
    log.msg((remote, local))
    lf = open(local, 'wb', 0)
    path = FilePath(self.currentDirectory).child(remote)
    d = self.client.openFile(path.path, filetransfer.FXF_READ, {})
    d.addCallback(self._cbGetOpenFile, lf)
    d.addErrback(self._ebCloseLf, lf)
    return d