import string
import struct
import twisted.internet.error
from twisted.conch import error
from twisted.conch.ssh import common, service
from twisted.internet import defer
from twisted.logger import Logger
from twisted.python.compat import nativeString, networkString
def sendExtendedData(self, channel, dataType, data):
    """
        Send extended data to a channel.  This should not normally be used:
        instead use channel.writeExtendedData(data, dataType) as it manages
        the window automatically.

        @type channel:  subclass of L{SSHChannel}
        @type dataType: L{int}
        @type data:     L{bytes}
        """
    if channel.localClosed:
        return
    self.transport.sendPacket(MSG_CHANNEL_EXTENDED_DATA, struct.pack('>2L', self.channelsToRemoteChannel[channel], dataType) + common.NS(data))