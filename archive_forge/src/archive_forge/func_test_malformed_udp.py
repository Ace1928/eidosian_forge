import unittest
import logging
import struct
from struct import *
from os_ken.ofproto import ether, inet
from os_ken.lib.packet.packet import Packet
from os_ken.lib.packet.udp import udp
from os_ken.lib.packet.ipv4 import ipv4
from os_ken.lib.packet import packet_utils
from os_ken.lib import addrconv
def test_malformed_udp(self):
    m_short_buf = self.buf[1:udp._MIN_LEN]
    self.assertRaises(Exception, udp.parser, m_short_buf)