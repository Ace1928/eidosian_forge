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
def lookupZone(self, name, timeout=10):
    address = self.pickServer()
    if address is None:
        return defer.fail(IOError('No domain name servers available'))
    host, port = address
    d = defer.Deferred()
    controller = AXFRController(name, d)
    factory = DNSClientFactory(controller, timeout)
    factory.noisy = False
    connector = self._reactor.connectTCP(host, port, factory)
    controller.timeoutCall = self._reactor.callLater(timeout or 10, self._timeoutZone, d, controller, connector, timeout or 10)

    def eliminateTimeout(failure):
        controller.timeoutCall.cancel()
        controller.timeoutCall = None
        return failure
    return d.addCallbacks(self._cbLookupZone, eliminateTimeout, callbackArgs=(connector,))