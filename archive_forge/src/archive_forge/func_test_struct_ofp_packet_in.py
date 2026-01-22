import unittest
import logging
from os_ken.ofproto.ofproto_v1_2 import *
def test_struct_ofp_packet_in(self):
    self.assertEqual(OFP_PACKET_IN_PACK_STR, '!IHBB')
    self.assertEqual(OFP_PACKET_IN_SIZE, 24)