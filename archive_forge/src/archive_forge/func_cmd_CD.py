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
def cmd_CD(self, path):
    path, rest = self._getFilename(path)
    if not path.endswith('/'):
        path += '/'
    newPath = path and os.path.join(self.currentDirectory, path) or ''
    d = self.client.openDirectory(newPath)
    d.addCallback(self._cbCd)
    d.addErrback(self._ebCommand)
    return d