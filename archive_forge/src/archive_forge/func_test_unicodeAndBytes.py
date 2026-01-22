from __future__ import annotations
from twisted.internet.abstract import isIPv6Address
from twisted.trial.unittest import SynchronousTestCase
def test_unicodeAndBytes(self) -> None:
    """
        L{isIPv6Address} evaluates ASCII-encoded bytes as well as text.
        """
    self.assertTrue(isIPv6Address(b'fe80::2%1'))
    self.assertTrue(isIPv6Address('fe80::2%1'))
    self.assertFalse(isIPv6Address('䌡'))
    self.assertFalse(isIPv6Address('hello%eth0'))
    self.assertFalse(isIPv6Address(b'hello%eth0'))