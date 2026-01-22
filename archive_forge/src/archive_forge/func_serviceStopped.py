import string
import struct
import twisted.internet.error
from twisted.conch import error
from twisted.conch.ssh import common, service
from twisted.internet import defer
from twisted.logger import Logger
from twisted.python.compat import nativeString, networkString
def serviceStopped(self):
    """
        Called when the connection is stopped.
        """
    for channel in list(self.channelsToRemoteChannel.keys()):
        self.channelClosed(channel)
    while self.channels:
        _, channel = self.channels.popitem()
        channel.openFailed(twisted.internet.error.ConnectionLost())
    self._cleanupGlobalDeferreds()