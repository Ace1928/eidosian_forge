import unittest
import logging
from os_ken.ofproto.ofproto_v1_2 import *
def test_enum_ofp_flow_mod_failed_code(self):
    self.assertEqual(OFPFMFC_UNKNOWN, 0)
    self.assertEqual(OFPFMFC_TABLE_FULL, 1)
    self.assertEqual(OFPFMFC_BAD_TABLE_ID, 2)
    self.assertEqual(OFPFMFC_OVERLAP, 3)
    self.assertEqual(OFPFMFC_EPERM, 4)
    self.assertEqual(OFPFMFC_BAD_TIMEOUT, 5)
    self.assertEqual(OFPFMFC_BAD_COMMAND, 6)
    self.assertEqual(OFPFMFC_BAD_FLAGS, 7)