import unittest
import logging
from os_ken.ofproto.ofproto_v1_2 import *
def test_struct_opf_queue_get_config_request(self):
    self.assertEqual(OFP_QUEUE_GET_CONFIG_REQUEST_PACK_STR, '!I4x')
    self.assertEqual(OFP_QUEUE_GET_CONFIG_REQUEST_SIZE, 16)