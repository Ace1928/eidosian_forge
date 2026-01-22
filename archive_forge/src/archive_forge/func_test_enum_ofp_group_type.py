import unittest
import logging
from os_ken.ofproto.ofproto_v1_2 import *
def test_enum_ofp_group_type(self):
    self.assertEqual(OFPGT_ALL, 0)
    self.assertEqual(OFPGT_SELECT, 1)
    self.assertEqual(OFPGT_INDIRECT, 2)
    self.assertEqual(OFPGT_FF, 3)