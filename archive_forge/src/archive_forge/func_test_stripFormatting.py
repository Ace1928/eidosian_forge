import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_stripFormatting(self):
    """
        Strip formatting codes from formatted text, leaving only the text parts.
        """
    self.assertEqual(irc.stripFormatting(irc.assembleFormattedText(A.bold[A.underline[A.reverseVideo[A.fg.red[A.bg.green['hello']]], ' world']])), 'hello world')