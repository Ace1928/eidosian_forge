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
def irc_LIST(self, prefix, params):
    """
        List query

        Return information about the indicated channels, or about all
        channels if none are specified.

        Parameters: [ <channel> *( "," <channel> ) [ <target> ] ]
        """
    if params:
        try:
            allChannels = params[0]
            if isinstance(allChannels, bytes):
                allChannels = allChannels.decode(self.encoding)
            channels = allChannels.split(',')
        except UnicodeDecodeError:
            self.sendMessage(irc.ERR_NOSUCHCHANNEL, params[0], ':No such channel (could not decode your unicode!)')
            return
        groups = []
        for ch in channels:
            if ch.startswith('#'):
                ch = ch[1:]
            groups.append(self.realm.lookupGroup(ch))
        groups = defer.DeferredList(groups, consumeErrors=True)
        groups.addCallback(lambda gs: [r for s, r in gs if s])
    else:
        groups = self.realm.itergroups()

    def cbGroups(groups):

        def gotSize(size, group):
            return (group.name, size, group.meta.get('topic'))
        d = defer.DeferredList([group.size().addCallback(gotSize, group) for group in groups])
        d.addCallback(lambda results: self.list([r for s, r in results if s]))
        return d
    groups.addCallback(cbGroups)