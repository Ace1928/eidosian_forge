import logging
import struct
import unittest
from os_ken.ofproto import ether
from os_ken.ofproto import inet
from os_ken.lib.packet import ethernet
from os_ken.lib.packet import packet
from os_ken.lib.packet import ipv4
from os_ken.lib.packet import vlan
from os_ken.lib.packet import pbb
def test_malformed_itag(self):
    m_short_buf = self.buf[1:pbb.itag._MIN_LEN]
    self.assertRaises(Exception, pbb.itag.parser, m_short_buf)