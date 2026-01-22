from twisted.internet import address
from twisted.trial import unittest
from .. import _v1parser
from .._exceptions import InvalidNetworkProtocol, InvalidProxyHeader, MissingAddressData
def test_fullParsingSuccess_IPv6(self) -> None:
    """
        Test that parsing is successful for an IPv6 PROXY header.
        """
    info = _v1parser.V1Parser.parse(b'PROXY TCP6 ::1 ::1 8080 8888')
    self.assertIsInstance(info.source, address.IPv6Address)
    assert isinstance(info.source, address.IPv6Address)
    assert isinstance(info.destination, address.IPv6Address)
    self.assertEqual(info.source.host, '::1')
    self.assertEqual(info.source.port, 8080)
    self.assertEqual(info.destination.host, '::1')
    self.assertEqual(info.destination.port, 8888)