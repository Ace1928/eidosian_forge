import unittest
import logging
from os_ken.ofproto.ofproto_v1_2 import *
def test_struct_ofp_flow_removed(self):
    self.assertEqual(OFP_FLOW_REMOVED_PACK_STR, '!QHBBIIHHQQHHBBBB')
    self.assertEqual(OFP_FLOW_REMOVED_PACK_STR0, '!QHBBIIHHQQ')
    self.assertEqual(OFP_FLOW_REMOVED_SIZE, 56)