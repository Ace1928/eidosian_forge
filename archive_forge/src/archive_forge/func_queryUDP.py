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
def queryUDP(self, queries, timeout=None):
    """
        Make a number of DNS queries via UDP.

        @type queries: A C{list} of C{dns.Query} instances
        @param queries: The queries to make.

        @type timeout: Sequence of C{int}
        @param timeout: Number of seconds after which to reissue the query.
        When the last timeout expires, the query is considered failed.

        @rtype: C{Deferred}
        @raise C{twisted.internet.defer.TimeoutError}: When the query times
        out.
        """
    if timeout is None:
        timeout = self.timeout
    addresses = self.servers + list(self.dynServers)
    if not addresses:
        return defer.fail(IOError('No domain name servers available'))
    addresses.reverse()
    used = addresses.pop()
    d = self._query(used, queries, timeout[0])
    d.addErrback(self._reissue, addresses, [used], queries, timeout)
    return d