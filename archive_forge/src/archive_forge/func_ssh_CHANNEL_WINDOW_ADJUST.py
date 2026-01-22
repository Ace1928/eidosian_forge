import string
import struct
import twisted.internet.error
from twisted.conch import error
from twisted.conch.ssh import common, service
from twisted.internet import defer
from twisted.logger import Logger
from twisted.python.compat import nativeString, networkString
def ssh_CHANNEL_WINDOW_ADJUST(self, packet):
    """
        The other side is adding bytes to its window.  Payload::
            uint32  local channel number
            uint32  bytes to add

        Call the channel's addWindowBytes() method to add new bytes to the
        remote window.
        """
    localChannel, bytesToAdd = struct.unpack('>2L', packet[:8])
    channel = self.channels[localChannel]
    channel.addWindowBytes(bytesToAdd)