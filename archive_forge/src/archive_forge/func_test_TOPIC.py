from twisted.cred import checkers, portal
from twisted.test import proto_helpers
from twisted.words.protocols import irc
from twisted.words.service import InMemoryWordsRealm, IRCFactory, IRCUser
from twisted.words.test.test_irc import IRCTestCase
def test_TOPIC(self):
    """
        Tests that irc_TOPIC sends ERR_NOSUCHCHANNEL if the channel name can't
        be decoded.
        """
    self.assertChokesOnBadBytes('TOPIC', irc.ERR_NOSUCHCHANNEL)