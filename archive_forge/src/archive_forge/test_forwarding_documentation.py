from twisted.python.reflect import requireModule
from twisted.internet.address import IPv6Address
from twisted.internet.test.test_endpoints import deterministicResolvingReactor
from twisted.internet.testing import MemoryReactorClock, StringTransport
from twisted.trial import unittest

        When a hostname is sent as part of forwarding requests, it
        is resolved using HostnameEndpoint's resolver.
        