from collections import defaultdict
from socket import (
from threading import Lock, local
from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted._threads import LockWorker, Team, createMemoryWorker
from twisted.internet._resolver import (
from twisted.internet.address import IPv4Address, IPv6Address
from twisted.internet.base import PluggableResolverMixin, ReactorBase
from twisted.internet.defer import Deferred
from twisted.internet.error import DNSLookupError
from twisted.internet.interfaces import (
from twisted.python.threadpool import ThreadPool
from twisted.trial.unittest import SynchronousTestCase as UnitTest
def test_resolveBoth(self):
    """
        When passed an C{addressTypes} parameter containing both L{IPv4Address}
        and L{IPv6Address} (or the default of C{None}, which carries the same
        meaning), L{GAIResolver} will pass C{AF_UNSPEC} to C{getaddrinfo}.
        """
    self._resolveOnlyTest([IPv4Address, IPv6Address], AF_UNSPEC)
    self._resolveOnlyTest(None, AF_UNSPEC)