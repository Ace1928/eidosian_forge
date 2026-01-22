import struct
from twisted.conch.ssh import channel
from twisted.conch.test import test_userauth
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.conch import error
def channel_TestChannel(self, windowSize, maxPacket, data):
    """
        The other side is requesting the TestChannel.  Create a C{TestChannel}
        instance, store it, and return it.
        """
    self.channel = TestChannel(remoteWindow=windowSize, remoteMaxPacket=maxPacket, data=data)
    return self.channel