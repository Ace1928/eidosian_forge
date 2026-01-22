from __future__ import annotations
from twisted.internet.abstract import isIPv6Address
from twisted.trial.unittest import SynchronousTestCase
def test_scopeID(self) -> None:
    """
        An otherwise valid IPv6 address literal may also include a C{"%"}
        followed by an arbitrary scope identifier.
        """
    self.assertTrue(isIPv6Address('fe80::1%eth0'))
    self.assertTrue(isIPv6Address('fe80::2%1'))
    self.assertTrue(isIPv6Address('fe80::3%en2'))