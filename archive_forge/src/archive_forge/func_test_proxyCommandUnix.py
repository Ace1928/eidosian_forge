from twisted.internet import address
from twisted.trial import unittest
from .. import _v2parser
from .._exceptions import InvalidProxyHeader
def test_proxyCommandUnix(self) -> None:
    """
        Test that proxy returns endpoint data for UNIX connections.
        """
    header = _makeHeaderUnix(verCom=b'!')
    info = _v2parser.V2Parser.parse(header)
    self.assertTrue(info.source)
    self.assertIsInstance(info.source, address.UNIXAddress)
    self.assertTrue(info.destination)
    self.assertIsInstance(info.destination, address.UNIXAddress)