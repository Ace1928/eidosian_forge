import string
import struct
import twisted.internet.error
from twisted.conch import error
from twisted.conch.ssh import common, service
from twisted.internet import defer
from twisted.logger import Logger
from twisted.python.compat import nativeString, networkString
def ssh_CHANNEL_REQUEST(self, packet):
    """
        The other side is sending a request to a channel.  Payload::
            uint32  local channel number
            string  request name
            bool    want reply
            <request specific data>

        Pass the message to the channel's requestReceived method.  If the
        other side wants a reply, add callbacks which will send the
        reply.
        """
    localChannel = struct.unpack('>L', packet[:4])[0]
    requestType, rest = common.getNS(packet[4:])
    wantReply = ord(rest[0:1])
    channel = self.channels[localChannel]
    d = defer.maybeDeferred(channel.requestReceived, requestType, rest[1:])
    if wantReply:
        d.addCallback(self._cbChannelRequest, localChannel)
        d.addErrback(self._ebChannelRequest, localChannel)
        return d