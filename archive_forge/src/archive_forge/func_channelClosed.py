import string
import struct
import twisted.internet.error
from twisted.conch import error
from twisted.conch.ssh import common, service
from twisted.internet import defer
from twisted.logger import Logger
from twisted.python.compat import nativeString, networkString
def channelClosed(self, channel):
    """
        Called when a channel is closed.
        It clears the local state related to the channel, and calls
        channel.closed().
        MAKE SURE YOU CALL THIS METHOD, even if you subclass L{SSHConnection}.
        If you don't, things will break mysteriously.

        @type channel: L{SSHChannel}
        """
    if channel in self.channelsToRemoteChannel:
        channel.localClosed = channel.remoteClosed = True
        del self.localToRemoteChannel[channel.id]
        del self.channels[channel.id]
        del self.channelsToRemoteChannel[channel]
        for d in self.deferreds.pop(channel.id, []):
            d.errback(error.ConchError('Channel closed.'))
        channel.closed()