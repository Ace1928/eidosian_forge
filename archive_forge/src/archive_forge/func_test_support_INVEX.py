import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_support_INVEX(self):
    """
        The INVEX support parameter is parsed into the mode character to be
        used for "invite exception" modes. If no parameter is specified then
        the character C{I} is assumed.
        """
    self.assertEqual(self._parseFeature('INVEX', 'Z'), 'Z')
    self.assertEqual(self._parseFeature('INVEX'), 'I')