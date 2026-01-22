import errno
import os
import warnings
from zope.interface import moduleProvides
from twisted.internet import defer, error, interfaces, protocol
from twisted.internet.abstract import isIPv6Address
from twisted.names import cache, common, dns, hosts as hostsModule, resolve, root
from twisted.python import failure, log
from twisted.python.compat import nativeString
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.internet.base import ThreadedResolver as _ThreadedResolverImpl
def createResolver(servers=None, resolvconf=None, hosts=None):
    """
    Create and return a Resolver.

    @type servers: C{list} of C{(str, int)} or L{None}

    @param servers: If not L{None}, interpreted as a list of domain name servers
    to attempt to use. Each server is a tuple of address in C{str} dotted-quad
    form and C{int} port number.

    @type resolvconf: C{str} or L{None}
    @param resolvconf: If not L{None}, on posix systems will be interpreted as
    an alternate resolv.conf to use. Will do nothing on windows systems. If
    L{None}, /etc/resolv.conf will be used.

    @type hosts: C{str} or L{None}
    @param hosts: If not L{None}, an alternate hosts file to use. If L{None}
    on posix systems, /etc/hosts will be used. On windows, C:\\windows\\hosts
    will be used.

    @rtype: C{IResolver}
    """
    if platform.getType() == 'posix':
        if resolvconf is None:
            resolvconf = b'/etc/resolv.conf'
        if hosts is None:
            hosts = b'/etc/hosts'
        theResolver = Resolver(resolvconf, servers)
        hostResolver = hostsModule.Resolver(hosts)
    else:
        if hosts is None:
            hosts = 'c:\\windows\\hosts'
        from twisted.internet import reactor
        bootstrap = _ThreadedResolverImpl(reactor)
        hostResolver = hostsModule.Resolver(hosts)
        theResolver = root.bootstrap(bootstrap, resolverFactory=Resolver)
    L = [hostResolver, cache.CacheResolver(), theResolver]
    return resolve.ResolverChain(L)