import unittest
import logging
from os_ken.ofproto.ofproto_v1_2 import *
def test_enum_ofp_match_type(self):
    self.assertEqual(OFPMT_STANDARD, 0)
    self.assertEqual(OFPMT_OXM, 1)