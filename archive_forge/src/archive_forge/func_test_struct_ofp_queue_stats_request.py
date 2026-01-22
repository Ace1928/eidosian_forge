import unittest
import logging
from os_ken.ofproto.ofproto_v1_2 import *
def test_struct_ofp_queue_stats_request(self):
    self.assertEqual(OFP_QUEUE_STATS_REQUEST_PACK_STR, '!II')
    self.assertEqual(OFP_QUEUE_STATS_REQUEST_SIZE, 8)