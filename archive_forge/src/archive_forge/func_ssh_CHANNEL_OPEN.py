import string
import struct
import twisted.internet.error
from twisted.conch import error
from twisted.conch.ssh import common, service
from twisted.internet import defer
from twisted.logger import Logger
from twisted.python.compat import nativeString, networkString
def ssh_CHANNEL_OPEN(self, packet):
    """
        The other side wants to get a channel.  Payload::
            string  channel name
            uint32  remote channel number
            uint32  remote window size
            uint32  remote maximum packet size
            <channel specific data>

        We get a channel from self.getChannel(), give it a local channel number
        and notify the other side.  Then notify the channel by calling its
        channelOpen method.
        """
    channelType, rest = common.getNS(packet)
    senderChannel, windowSize, maxPacket = struct.unpack('>3L', rest[:12])
    packet = rest[12:]
    try:
        channel = self.getChannel(channelType, windowSize, maxPacket, packet)
        localChannel = self.localChannelID
        self.localChannelID += 1
        channel.id = localChannel
        self.channels[localChannel] = channel
        self.channelsToRemoteChannel[channel] = senderChannel
        self.localToRemoteChannel[localChannel] = senderChannel
        openConfirmPacket = struct.pack('>4L', senderChannel, localChannel, channel.localWindowSize, channel.localMaxPacket) + channel.specificData
        self.transport.sendPacket(MSG_CHANNEL_OPEN_CONFIRMATION, openConfirmPacket)
        channel.channelOpen(packet)
    except Exception as e:
        self._log.failure('channel open failed')
        if isinstance(e, error.ConchError):
            textualInfo, reason = e.args
            if isinstance(textualInfo, int):
                textualInfo, reason = (reason, textualInfo)
        else:
            reason = OPEN_CONNECT_FAILED
            textualInfo = 'unknown failure'
        self.transport.sendPacket(MSG_CHANNEL_OPEN_FAILURE, struct.pack('>2L', senderChannel, reason) + common.NS(networkString(textualInfo)) + common.NS(b''))