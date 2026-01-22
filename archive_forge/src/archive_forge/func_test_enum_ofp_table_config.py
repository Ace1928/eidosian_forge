import unittest
import logging
from os_ken.ofproto.ofproto_v1_2 import *
def test_enum_ofp_table_config(self):
    self.assertEqual(OFPTC_TABLE_MISS_CONTROLLER, 0)
    self.assertEqual(OFPTC_TABLE_MISS_CONTINUE, 1 << 0)
    self.assertEqual(OFPTC_TABLE_MISS_DROP, 1 << 1)
    self.assertEqual(OFPTC_TABLE_MISS_MASK, 3)