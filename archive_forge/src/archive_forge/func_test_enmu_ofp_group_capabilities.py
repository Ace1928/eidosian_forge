import unittest
import logging
from os_ken.ofproto.ofproto_v1_2 import *
def test_enmu_ofp_group_capabilities(self):
    self.assertEqual(OFPGFC_SELECT_WEIGHT, 1 << 0)
    self.assertEqual(OFPGFC_SELECT_LIVENESS, 1 << 1)
    self.assertEqual(OFPGFC_CHAINING, 1 << 2)
    self.assertEqual(OFPGFC_CHAINING_CHECKS, 1 << 3)