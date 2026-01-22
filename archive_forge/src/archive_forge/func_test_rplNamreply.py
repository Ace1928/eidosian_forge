from twisted.internet.testing import StringTransport
from twisted.words.im.basechat import ChatUI, Conversation, GroupConversation
from twisted.words.im.ircsupport import IRCAccount, IRCProto
from twisted.words.im.locals import OfflineError
from twisted.words.test.test_irc import IRCTestCase
def test_rplNamreply(self) -> None:
    """
        RPL_NAMREPLY server response (353) lists all the users in a channel.
        RPL_ENDOFNAMES server response (363) is sent at the end of RPL_NAMREPLY
        to indicate that there are no more names.
        """
    self.proto.makeConnection(self.transport)
    self.proto.dataReceived(':example.com 353 z3p = #bnl :pSwede Dan- SkOyg @MrOp +MrPlus\r\n')
    expectedInGroups = {'Dan-': ['bnl'], 'pSwede': ['bnl'], 'SkOyg': ['bnl'], 'MrOp': ['bnl'], 'MrPlus': ['bnl']}
    expectedNamReplies = {'bnl': ['pSwede', 'Dan-', 'SkOyg', 'MrOp', 'MrPlus']}
    self.assertEqual(expectedInGroups, self.proto._ingroups)
    self.assertEqual(expectedNamReplies, self.proto._namreplies)
    self.proto.dataReceived(':example.com 366 alice #bnl :End of /NAMES list\r\n')
    self.assertEqual({}, self.proto._namreplies)
    groupConversation = self.proto.getGroupConversation('bnl')
    self.assertEqual(expectedNamReplies['bnl'], groupConversation.members)