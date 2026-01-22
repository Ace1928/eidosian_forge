from twisted.internet import address
from twisted.trial import unittest
from .. import _v1parser
from .._exceptions import InvalidNetworkProtocol, InvalidProxyHeader, MissingAddressData
def test_feedParsingTooLong(self) -> None:
    """
        Test that parsing fails if no newline is found in 108 bytes.
        """
    parser = _v1parser.V1Parser()
    info, remaining = parser.feed(b'PROXY TCP4 127.0.0.1 127.0.0.1 ')
    self.assertFalse(info)
    self.assertFalse(remaining)
    info, remaining = parser.feed(b'8080 8888')
    self.assertFalse(info)
    self.assertFalse(remaining)
    self.assertRaises(InvalidProxyHeader, parser.feed, b' ' * 100)