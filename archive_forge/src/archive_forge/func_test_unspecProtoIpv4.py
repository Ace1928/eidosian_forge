from twisted.internet import address
from twisted.trial import unittest
from .. import _v2parser
from .._exceptions import InvalidProxyHeader
def test_unspecProtoIpv4(self) -> None:
    """
        Test that UNSPEC does not return endpoint data for IPv4 connections.
        """
    header = _makeHeaderIPv4(famProto=b'\x10')
    info = _v2parser.V2Parser.parse(header)
    self.assertFalse(info.source)
    self.assertFalse(info.destination)