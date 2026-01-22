import unittest
import logging
from os_ken.ofproto.ofproto_v1_2 import *
def test_struct_ofp_bucket_counter(self):
    self.assertEqual(OFP_BUCKET_COUNTER_PACK_STR, '!QQ')
    self.assertEqual(OFP_BUCKET_COUNTER_SIZE, 16)