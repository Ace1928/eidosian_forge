from time import ctime, time
from zope.interface import implementer
from twisted import copyright
from twisted.cred import credentials, error as ecred, portal
from twisted.internet import defer, protocol
from twisted.python import failure, log, reflect
from twisted.python.components import registerAdapter
from twisted.spread import pb
from twisted.words import ewords, iwords
from twisted.words.protocols import irc
def irc_WHO(self, prefix, params):
    """
        Who query

        Parameters: [ <mask> [ "o" ] ]
        """
    if not params:
        self.sendMessage(irc.RPL_ENDOFWHO, ':/WHO not supported.')
        return
    try:
        channelOrUser = params[0]
        if isinstance(channelOrUser, bytes):
            channelOrUser = channelOrUser.decode(self.encoding)
    except UnicodeDecodeError:
        self.sendMessage(irc.RPL_ENDOFWHO, params[0], ':End of /WHO list (could not decode your unicode!)')
        return
    if channelOrUser.startswith('#'):

        def ebGroup(err):
            err.trap(ewords.NoSuchGroup)
            self.sendMessage(irc.RPL_ENDOFWHO, channelOrUser, ':End of /WHO list.')
        d = self.realm.lookupGroup(channelOrUser[1:])
        d.addCallbacks(self._channelWho, ebGroup)
    else:

        def ebUser(err):
            err.trap(ewords.NoSuchUser)
            self.sendMessage(irc.RPL_ENDOFWHO, channelOrUser, ':End of /WHO list.')
        d = self.realm.lookupUser(channelOrUser)
        d.addCallbacks(self._userWho, ebUser)