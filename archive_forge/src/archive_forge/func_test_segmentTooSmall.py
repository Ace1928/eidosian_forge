from twisted.internet import address
from twisted.trial import unittest
from .. import _v2parser
from .._exceptions import InvalidProxyHeader
def test_segmentTooSmall(self) -> None:
    """
        Test that an initial payload of less than 16 bytes fails.
        """
    testValue = b'NEEDMOREDATA'
    parser = _v2parser.V2Parser()
    self.assertRaises(InvalidProxyHeader, parser.feed, testValue)