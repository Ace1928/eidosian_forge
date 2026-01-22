import unittest
import logging
import inspect
import struct
from os_ken.lib import addrconv
from os_ken.lib import ip
from os_ken.lib.packet import ipv6
def test_len_re(self):
    size = 5
    auth = ipv6.auth(0, size, 256, 1, b'!\xd3\xa9\\_\xfdM\x18F"\xb9\xf8\xf8\xf8\xf8\xf8')
    self.assertEqual((size + 2) * 4, len(auth))