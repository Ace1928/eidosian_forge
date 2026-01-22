import unittest
import logging
from os_ken.ofproto.ofproto_v1_2 import *
def test_struct_ofp_queue_get_config_reply(self):
    self.assertEqual(OFP_QUEUE_GET_CONFIG_REPLY_PACK_STR, '!I4x')
    self.assertEqual(OFP_QUEUE_GET_CONFIG_REPLY_SIZE, 16)