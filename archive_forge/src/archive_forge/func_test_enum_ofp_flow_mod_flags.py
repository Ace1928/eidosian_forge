import unittest
import logging
from os_ken.ofproto.ofproto_v1_2 import *
def test_enum_ofp_flow_mod_flags(self):
    self.assertEqual(OFPFF_SEND_FLOW_REM, 1 << 0)
    self.assertEqual(OFPFF_CHECK_OVERLAP, 1 << 1)
    self.assertEqual(OFPFF_RESET_COUNTS, 1 << 2)