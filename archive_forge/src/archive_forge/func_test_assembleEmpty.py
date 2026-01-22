import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_assembleEmpty(self):
    """
        An attribute with no text assembles to the empty string. An attribute
        whose text is the empty string assembles to two control codes: C{off}
        and that of the attribute.
        """
    self.assertEqual(irc.assembleFormattedText(A.normal), '')
    self.assertEqual(irc.assembleFormattedText(A.bold['']), '\x0f\x02')