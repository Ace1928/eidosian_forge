import unittest
import logging
from os_ken.ofproto.ofproto_v1_2 import *
def test_enum_ofp_stats_types(self):
    self.assertEqual(OFPST_DESC, 0)
    self.assertEqual(OFPST_FLOW, 1)
    self.assertEqual(OFPST_AGGREGATE, 2)
    self.assertEqual(OFPST_TABLE, 3)
    self.assertEqual(OFPST_PORT, 4)
    self.assertEqual(OFPST_QUEUE, 5)
    self.assertEqual(OFPST_GROUP, 6)
    self.assertEqual(OFPST_GROUP_DESC, 7)
    self.assertEqual(OFPST_GROUP_FEATURES, 8)
    self.assertEqual(OFPST_EXPERIMENTER, 65535)