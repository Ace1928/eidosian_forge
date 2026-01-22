import unittest
import logging
import socket
from struct import *
from os_ken.ofproto.ofproto_v1_2_parser import *
from os_ken.ofproto import ofproto_v1_2_parser
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ofproto_protocol
from os_ken.ofproto import ether
from os_ken.ofproto.ofproto_parser import MsgBase
from os_ken import utils
from os_ken.lib import addrconv
from os_ken.lib import pack_utils
def test_set_ipv6_dst_mask_mid(self):
    ipv6 = 'e9e8:9ea5:7d67:82cc:ca54:1fc0:2d24:f038'
    mask = ':'.join(['ffff'] * 4 + ['0'] * 4)
    self._test_set_ipv6_dst(ipv6, mask)