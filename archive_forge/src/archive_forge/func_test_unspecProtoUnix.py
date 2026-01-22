from twisted.internet import address
from twisted.trial import unittest
from .. import _v2parser
from .._exceptions import InvalidProxyHeader
def test_unspecProtoUnix(self) -> None:
    """
        Test that UNSPEC does not return endpoint data for UNIX connections.
        """
    header = _makeHeaderUnix(famProto=b'0')
    info = _v2parser.V2Parser.parse(header)
    self.assertFalse(info.source)
    self.assertFalse(info.destination)