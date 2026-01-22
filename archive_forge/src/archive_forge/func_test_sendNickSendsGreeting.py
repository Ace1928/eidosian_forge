from twisted.cred import checkers, portal
from twisted.test import proto_helpers
from twisted.words.protocols import irc
from twisted.words.service import InMemoryWordsRealm, IRCFactory, IRCUser
from twisted.words.test.test_irc import IRCTestCase
def test_sendNickSendsGreeting(self):
    """
        Receiving NICK without authenticating sends the MOTD Start and MOTD End
        messages, which is required by certain popular IRC clients (such as
        Pidgin) before a connection is considered to be fully established.
        """
    self.ircUser.irc_NICK('', ['mynick'])
    response = self.response()
    start = list(self.scanResponse(response, irc.RPL_MOTDSTART))
    end = list(self.scanResponse(response, irc.RPL_ENDOFMOTD))
    self.assertEqual(start, [(0, ('example.com', '375', ['mynick', '- example.com Message of the Day - ']))])
    self.assertEqual(end, [(1, ('example.com', '376', ['mynick', 'End of /MOTD command.']))])