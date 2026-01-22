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
def cmd_CHMOD(self, rest):
    mod, rest = rest.split(None, 1)
    path, rest = self._getFilename(rest)
    mod = int(mod, 8)
    d = self.client.setAttrs(path, {'permissions': mod})
    d.addCallback(_ignore)
    return d