import unittest
import logging
from os_ken.ofproto.ofproto_v1_2 import *
def test_struct_ofp_port_stats_request(self):
    self.assertEqual(OFP_PORT_STATS_REQUEST_PACK_STR, '!I4x')
    self.assertEqual(OFP_PORT_STATS_REQUEST_SIZE, 8)