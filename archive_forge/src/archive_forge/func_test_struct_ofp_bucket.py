import unittest
import logging
from os_ken.ofproto.ofproto_v1_2 import *
def test_struct_ofp_bucket(self):
    self.assertEqual(OFP_BUCKET_PACK_STR, '!HHII4x')
    self.assertEqual(OFP_BUCKET_SIZE, 16)