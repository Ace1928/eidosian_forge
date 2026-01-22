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
def packRequest_pty_req(term, geometry, modes):
    """
    Pack a pty-req request so that it is suitable for sending.

    NOTE: modes must be packed before being sent here.

    @type geometry: L{tuple}
    @param geometry: A tuple of (rows, columns, xpixel, ypixel)
    """
    rows, cols, xpixel, ypixel = geometry
    termPacked = common.NS(term)
    winSizePacked = struct.pack('>4L', cols, rows, xpixel, ypixel)
    modesPacked = common.NS(modes)
    return termPacked + winSizePacked + modesPacked