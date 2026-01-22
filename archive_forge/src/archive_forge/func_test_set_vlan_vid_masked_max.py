import unittest
import logging
import socket
from struct import *
from os_ken.ofproto.ofproto_v1_3_parser import *
from os_ken.ofproto import ofproto_v1_3
from os_ken.ofproto import ofproto_protocol
def test_set_vlan_vid_masked_max(self):
    self._test_set_vlan_vid(2047, 4095)