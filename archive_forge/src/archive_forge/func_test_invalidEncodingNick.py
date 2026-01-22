from twisted.cred import checkers, portal
from twisted.test import proto_helpers
from twisted.words.protocols import irc
from twisted.words.service import InMemoryWordsRealm, IRCFactory, IRCUser
from twisted.words.test.test_irc import IRCTestCase
def test_invalidEncodingNick(self):
    """
        A NICK command sent with a nickname that cannot be decoded with the
        current IRCUser's encoding results in a PRIVMSG from NickServ
        indicating that the nickname could not be decoded.
        """
    self.ircUser.irc_NICK('', [b'\xd4\xc5\xd3\xd4'])
    self.assertRaises(UnicodeError)