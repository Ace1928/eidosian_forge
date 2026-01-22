import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_multipleLine_notice(self):
    """
        Messages longer than the C{length} parameter to L{IRCClient.notice}
        will be split and sent in multiple commands.
        """
    maxLen_notice = len('NOTICE foo :') + 3 + 2
    self.client.notice('foo', 'barbazbo', maxLen_notice)
    self.assertEqual(self.client.lines, ['NOTICE foo :bar', 'NOTICE foo :baz', 'NOTICE foo :bo'])