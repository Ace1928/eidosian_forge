import unittest
import platform
import sys
from os_ken.lib import sockaddr
def test_sockaddr_linux_sa_in4(self):
    if system != 'Linux' or sys.byteorder != 'little':
        return
    addr = '127.0.0.1'
    expected_result = b'\x02\x00\x00\x00\x7f\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00'
    self.assertEqual(expected_result, sockaddr.sa_in4(addr))