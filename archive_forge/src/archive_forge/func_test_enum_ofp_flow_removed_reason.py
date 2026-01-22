import unittest
import logging
from os_ken.ofproto.ofproto_v1_2 import *
def test_enum_ofp_flow_removed_reason(self):
    self.assertEqual(OFPRR_IDLE_TIMEOUT, 0)
    self.assertEqual(OFPRR_HARD_TIMEOUT, 1)
    self.assertEqual(OFPRR_DELETE, 2)
    self.assertEqual(OFPRR_GROUP_DELETE, 3)