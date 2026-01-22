import unittest
import logging
from os_ken.ofproto.ofproto_v1_2 import *
def test_struct_ofp_action_experimenter_header(self):
    self.assertEqual(OFP_ACTION_EXPERIMENTER_HEADER_PACK_STR, '!HHI')
    self.assertEqual(OFP_ACTION_EXPERIMENTER_HEADER_SIZE, 8)