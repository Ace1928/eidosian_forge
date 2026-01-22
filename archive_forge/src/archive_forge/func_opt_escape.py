import fcntl
import getpass
import os
import signal
import struct
import sys
import tty
from typing import List, Tuple
from twisted.conch.client import connect, default
from twisted.conch.client.options import ConchOptions
from twisted.conch.error import ConchError
from twisted.conch.ssh import channel, common, connection, forwarding, session
from twisted.internet import reactor, stdio, task
from twisted.python import log, usage
from twisted.python.compat import ioType, networkString
def opt_escape(self, esc):
    """
        Set escape character; ``none'' = disable
        """
    if esc == 'none':
        self['escape'] = None
    elif esc[0] == '^' and len(esc) == 2:
        self['escape'] = chr(ord(esc[1]) - 64)
    elif len(esc) == 1:
        self['escape'] = esc
    else:
        sys.exit(f"Bad escape character '{esc}'.")