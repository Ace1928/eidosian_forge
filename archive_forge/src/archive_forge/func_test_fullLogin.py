from twisted.cred import checkers, portal
from twisted.test import proto_helpers
from twisted.words.protocols import irc
from twisted.words.service import InMemoryWordsRealm, IRCFactory, IRCUser
from twisted.words.test.test_irc import IRCTestCase
def test_fullLogin(self):
    """
        Receiving USER, PASS, NICK will log in the user, and transmit the
        appropriate response messages.
        """
    self.ircUser.irc_USER('', ['john doe'])
    self.ircUser.irc_PASS('', ['pass'])
    self.ircUser.irc_NICK('', ['john'])
    version = 'Your host is example.com, running version {}'.format(self.factory._serverInfo['serviceVersion'])
    creation = 'This server was created on {}'.format(self.factory._serverInfo['creationDate'])
    self.assertEqual(self.response(), [('example.com', '375', ['john', '- example.com Message of the Day - ']), ('example.com', '376', ['john', 'End of /MOTD command.']), ('example.com', '001', ['john', 'connected to Twisted IRC']), ('example.com', '002', ['john', version]), ('example.com', '003', ['john', creation]), ('example.com', '004', ['john', 'example.com', self.factory._serverInfo['serviceVersion'], 'w', 'n'])])