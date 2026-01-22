from twisted.internet import address
from twisted.trial import unittest
from .. import _v2parser
from .._exceptions import InvalidProxyHeader
def test_overflowIpv6(self) -> None:
    """
        Test that overflow bits are preserved during feed parsing for IPv6.
        """
    testValue = b'TEST DATA\r\n\r\nTEST DATA'
    header = _makeHeaderIPv6() + testValue
    parser = _v2parser.V2Parser()
    info, overflow = parser.feed(header)
    self.assertTrue(info)
    self.assertEqual(overflow, testValue)