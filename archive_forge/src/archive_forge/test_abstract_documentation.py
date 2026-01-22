from __future__ import annotations
from twisted.internet.abstract import isIPv6Address
from twisted.trial.unittest import SynchronousTestCase

        L{isIPv6Address} evaluates ASCII-encoded bytes as well as text.
        