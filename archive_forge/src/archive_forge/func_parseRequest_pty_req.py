import os
import signal
import struct
import sys
from zope.interface import implementer
from twisted.conch.interfaces import (
from twisted.conch.ssh import channel, common, connection
from twisted.internet import interfaces, protocol
from twisted.logger import Logger
from twisted.python.compat import networkString
def parseRequest_pty_req(data):
    """Parse the data from a pty-req request into usable data.

    @returns: a tuple of (terminal type, (rows, cols, xpixel, ypixel), modes)
    """
    term, rest = common.getNS(data)
    cols, rows, xpixel, ypixel = struct.unpack('>4L', rest[:16])
    modes, ignored = common.getNS(rest[16:])
    winSize = (rows, cols, xpixel, ypixel)
    modes = [(ord(modes[i:i + 1]), struct.unpack('>L', modes[i + 1:i + 5])[0]) for i in range(0, len(modes) - 1, 5)]
    return (term, winSize, modes)