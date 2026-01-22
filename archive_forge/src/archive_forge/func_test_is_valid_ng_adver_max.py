import unittest
import logging
import struct
import inspect
from os_ken.ofproto import inet
from os_ken.lib.packet import ipv4
from os_ken.lib.packet import ipv6
from os_ken.lib.packet import packet
from os_ken.lib.packet import packet_utils
from os_ken.lib.packet import vrrp
from os_ken.lib import addrconv
def test_is_valid_ng_adver_max(self):
    max_adver_int = vrrp.VRRP_V3_MAX_ADVER_INT_MAX + 1
    self.assertTrue(not self._test_is_valid(max_adver_int=max_adver_int))