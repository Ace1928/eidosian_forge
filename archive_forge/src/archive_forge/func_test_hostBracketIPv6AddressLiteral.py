from urllib.parse import urlparse
from twisted.trial import unittest
from twisted.web import client
def test_hostBracketIPv6AddressLiteral(self):
    """
        Brackets around IPv6 addresses are stripped in the host field. The host
        field is then exported with brackets in the output of
        L{client.URI.toBytes}.
        """
    uri = client.URI.fromBytes(b'http://[::1]:80/index.html')
    self.assertEqual(uri.host, b'::1')
    self.assertEqual(uri.netloc, b'[::1]:80')
    self.assertEqual(uri.toBytes(), b'http://[::1]:80/index.html')