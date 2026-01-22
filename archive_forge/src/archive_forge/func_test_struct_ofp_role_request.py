import unittest
import logging
from os_ken.ofproto.ofproto_v1_2 import *
def test_struct_ofp_role_request(self):
    self.assertEqual(OFP_ROLE_REQUEST_PACK_STR, '!I4xQ')
    self.assertEqual(OFP_ROLE_REQUEST_SIZE, 24)