from twisted.cred import checkers, portal
from twisted.test import proto_helpers
from twisted.words.protocols import irc
from twisted.words.service import InMemoryWordsRealm, IRCFactory, IRCUser
from twisted.words.test.test_irc import IRCTestCase
def test_PART(self):
    """
        Tests that irc_PART sends ERR_NOTONCHANNEL if the target name can't
        be decoded.
        """
    self.assertChokesOnBadBytes('PART', irc.ERR_NOTONCHANNEL)