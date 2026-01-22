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
def cmd_EXEC(self, rest):
    """
        Run C{rest} using the user's shell (or /bin/sh if they do not have
        one).
        """
    shell = self._pwd.getpwnam(getpass.getuser())[6]
    if not shell:
        shell = '/bin/sh'
    if rest:
        cmds = ['-c', rest]
        return utils.getProcessOutput(shell, cmds, errortoo=1)
    else:
        os.system(shell)