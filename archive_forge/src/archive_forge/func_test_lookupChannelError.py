import struct
from twisted.conch.ssh import channel
from twisted.conch.test import test_userauth
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.conch import error
def test_lookupChannelError(self):
    """
        If a C{lookupChannel} implementation raises L{error.ConchError} with the
        arguments in the wrong order, a C{MSG_CHANNEL_OPEN} failure is still
        sent in response to the message.

        This is a temporary work-around until L{error.ConchError} is given
        better attributes and all of the Conch code starts constructing
        instances of it properly.  Eventually this functionality should be
        deprecated and then removed.
        """
    self._lookupChannelErrorTest(123)