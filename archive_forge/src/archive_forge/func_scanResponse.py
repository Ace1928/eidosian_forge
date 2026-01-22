from twisted.cred import checkers, portal
from twisted.test import proto_helpers
from twisted.words.protocols import irc
from twisted.words.service import InMemoryWordsRealm, IRCFactory, IRCUser
from twisted.words.test.test_irc import IRCTestCase
def scanResponse(self, response, messageType):
    """
        Gets messages out of a response

        @param response: The parsed IRC messages of the response, as returned
        by L{IRCUserTests.response}

        @param messageType: The string type of the desired messages.

        @return: An iterator which yields 2-tuples of C{(index, ircMessage)}
        """
    for n, message in enumerate(response):
        if message[1] == messageType:
            yield (n, message)