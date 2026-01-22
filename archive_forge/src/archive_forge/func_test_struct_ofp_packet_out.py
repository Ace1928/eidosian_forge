import unittest
import logging
from os_ken.ofproto.ofproto_v1_2 import *
def test_struct_ofp_packet_out(self):
    self.assertEqual(OFP_PACKET_OUT_PACK_STR, '!IIH6x')
    self.assertEqual(OFP_PACKET_OUT_SIZE, 24)