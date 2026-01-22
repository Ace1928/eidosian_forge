import unittest
import logging
from os_ken.ofproto.ofproto_v1_2 import *
def test_struct_ofp_port_status(self):
    self.assertEqual(OFP_PORT_STATUS_PACK_STR, '!B7xI4x6s2x16sIIIIIIII')
    self.assertEqual(OFP_PORT_STATUS_DESC_OFFSET, 16)
    self.assertEqual(OFP_PORT_STATUS_SIZE, 80)