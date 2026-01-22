import unittest
import logging
from os_ken.ofproto.ofproto_v1_2 import *
def test_struct_ofp_flow_stats(self):
    self.assertEqual(OFP_FLOW_STATS_PACK_STR, '!HBxIIHHH6xQQQ')
    self.assertEqual(OFP_FLOW_STATS_SIZE, 56)