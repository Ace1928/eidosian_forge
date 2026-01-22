import unittest
import logging
from os_ken.ofproto.ofproto_v1_2 import *
def test_struct_ofp_desc_stats(self):
    self.assertEqual(OFP_DESC_STATS_PACK_STR, '!256s256s256s32s256s')
    self.assertEqual(OFP_DESC_STATS_SIZE, 1056)