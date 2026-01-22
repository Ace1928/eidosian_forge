import unittest
import logging
from os_ken.ofproto.ofproto_v1_2 import *
def test_enum_ofp_table_mod_failed_code(self):
    self.assertEqual(OFPTMFC_BAD_TABLE, 0)
    self.assertEqual(OFPTMFC_BAD_CONFIG, 1)
    self.assertEqual(OFPTMFC_EPERM, 2)