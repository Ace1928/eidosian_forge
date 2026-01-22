from twisted.internet.testing import StringTransport
from twisted.words.im.basechat import ChatUI, Conversation, GroupConversation
from twisted.words.im.ircsupport import IRCAccount, IRCProto
from twisted.words.im.locals import OfflineError
from twisted.words.test.test_irc import IRCTestCase
def test_authenticate(self) -> None:
    """
        If created with an account with a password, L{IRCProto} sends a
        I{PASS} command before the I{NICK} and I{USER} commands.
        """
    self.account.password = 'secret'
    self.proto.makeConnection(self.transport)
    self.assertEqualBufferValue(self.transport.value(), 'PASS secret\r\nNICK alice\r\nUSER alice foo bar :Twisted-IM user\r\n')