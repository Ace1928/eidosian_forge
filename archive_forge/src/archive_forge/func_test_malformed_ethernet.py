import unittest
import logging
import struct
from struct import *
from os_ken.ofproto import ether, inet
from os_ken.lib.packet.ethernet import ethernet
from os_ken.lib.packet.arp import arp
from os_ken.lib import addrconv
def test_malformed_ethernet(self):
    m_short_buf = self.buf[1:ethernet._MIN_LEN]
    self.assertRaises(Exception, ethernet.parser, m_short_buf)