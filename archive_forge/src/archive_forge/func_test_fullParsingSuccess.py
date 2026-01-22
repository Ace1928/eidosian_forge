from twisted.internet import address
from twisted.trial import unittest
from .. import _v1parser
from .._exceptions import InvalidNetworkProtocol, InvalidProxyHeader, MissingAddressData
def test_fullParsingSuccess(self) -> None:
    """
        Test that parsing is successful for a PROXY header.
        """
    info = _v1parser.V1Parser.parse(b'PROXY TCP4 127.0.0.1 127.0.0.1 8080 8888')
    self.assertIsInstance(info.source, address.IPv4Address)
    assert isinstance(info.source, address.IPv4Address)
    assert isinstance(info.destination, address.IPv4Address)
    self.assertEqual(info.source.host, '127.0.0.1')
    self.assertEqual(info.source.port, 8080)
    self.assertEqual(info.destination.host, '127.0.0.1')
    self.assertEqual(info.destination.port, 8888)