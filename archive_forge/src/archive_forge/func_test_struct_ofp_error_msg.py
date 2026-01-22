import unittest
import logging
from os_ken.ofproto.ofproto_v1_2 import *
def test_struct_ofp_error_msg(self):
    self.assertEqual(OFP_ERROR_MSG_PACK_STR, '!HH')
    self.assertEqual(OFP_ERROR_MSG_SIZE, 12)