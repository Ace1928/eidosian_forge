import unittest
import logging
from os_ken.ofproto.ofproto_v1_2 import *
def test_enum_ofp_group(self):
    self.assertEqual(OFPG_MAX, 4294967040)
    self.assertEqual(OFPG_ALL, 4294967292)
    self.assertEqual(OFPG_ANY, 4294967295)