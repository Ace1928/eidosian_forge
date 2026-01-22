import unittest
import logging
from os_ken.ofproto.ofproto_v1_2 import *
def test_struct_ofp_group_features_stats(self):
    self.assertEqual(OFP_GROUP_FEATURES_STATS_PACK_STR, '!II4I4I')
    self.assertEqual(OFP_GROUP_FEATURES_STATS_SIZE, 40)