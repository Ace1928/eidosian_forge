from twisted.internet.testing import StringTransport
from twisted.words.im.basechat import ChatUI, Conversation, GroupConversation
from twisted.words.im.ircsupport import IRCAccount, IRCProto
from twisted.words.im.locals import OfflineError
from twisted.words.test.test_irc import IRCTestCase
def test_login(self) -> None:
    """
        When L{IRCProto} is connected to a transport, it sends I{NICK} and
        I{USER} commands with the username from the account object.
        """
    self.proto.makeConnection(self.transport)
    self.assertEqualBufferValue(self.transport.value(), 'NICK alice\r\nUSER alice foo bar :Twisted-IM user\r\n')