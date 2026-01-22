import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_support_EXCEPTS(self):
    """
        The EXCEPTS support parameter is parsed into the mode character
        to be used for "ban exception" modes. If no parameter is specified
        then the character C{e} is assumed.
        """
    self.assertEqual(self._parseFeature('EXCEPTS', 'Z'), 'Z')
    self.assertEqual(self._parseFeature('EXCEPTS'), 'e')