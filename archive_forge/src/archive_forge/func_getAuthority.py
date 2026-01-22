from twisted.application import service
from twisted.internet import defer, task
from twisted.names import client, common, dns, resolve
from twisted.names.authority import FileAuthority
from twisted.python import failure, log
from twisted.python.compat import nativeString
def getAuthority(self):
    """
        Get a resolver for the transferred domains.

        @rtype: L{ResolverChain}
        """
    return resolve.ResolverChain(self.domains)