import unittest
import platform
import sys
from os_ken.lib import sockaddr
def test_sockaddr_linux_sa_in6(self):
    if system != 'Linux' or sys.byteorder != 'little':
        return
    addr = 'dead:beef::1'
    expected_result = b'\n\x00\x00\x00\x00\x00\x00\x00\xde\xad\xbe\xef\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00'
    self.assertEqual(expected_result, sockaddr.sa_in6(addr))