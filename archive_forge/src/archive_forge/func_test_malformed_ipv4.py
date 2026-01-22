import unittest
import logging
import struct
from struct import *
from os_ken.ofproto import inet
from os_ken.lib.packet import packet_utils
from os_ken.lib.packet.ipv4 import ipv4
from os_ken.lib.packet.tcp import tcp
from os_ken.lib import addrconv
def test_malformed_ipv4(self):
    m_short_buf = self.buf[1:ipv4._MIN_LEN]
    self.assertRaises(Exception, ipv4.parser, m_short_buf)