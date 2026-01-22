import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_support_STATUSMSG(self):
    """
        The STATUSMSG support parameter is parsed into a string of channel
        status that support the exclusive channel notice method.
        """
    self.assertEqual(self._parseFeature('STATUSMSG', '@+'), '@+')