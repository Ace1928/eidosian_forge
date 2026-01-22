import unittest
import logging
from os_ken.ofproto.ofproto_v1_2 import *
def test_struct_ofp_action_set_queue(self):
    self.assertEqual(OFP_ACTION_SET_QUEUE_PACK_STR, '!HHI')
    self.assertEqual(OFP_ACTION_SET_QUEUE_SIZE, 8)