import unittest
import logging
from os_ken.ofproto.ofproto_v1_2 import *
def test_struct_ofp_switch_config(self):
    self.assertEqual(OFP_SWITCH_CONFIG_PACK_STR, '!HH')
    self.assertEqual(OFP_SWITCH_CONFIG_SIZE, 12)