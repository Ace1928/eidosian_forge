import unittest
import logging
from os_ken.ofproto.ofproto_v1_2 import *
def test_struct_ofp_queue_prop_max_rate(self):
    self.assertEqual(OFP_QUEUE_PROP_MAX_RATE_PACK_STR, '!H6x')
    self.assertEqual(OFP_QUEUE_PROP_MAX_RATE_SIZE, 16)