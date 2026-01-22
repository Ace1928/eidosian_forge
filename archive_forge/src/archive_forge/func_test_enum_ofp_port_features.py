import unittest
import logging
from os_ken.ofproto.ofproto_v1_2 import *
def test_enum_ofp_port_features(self):
    self.assertEqual(OFPPF_10MB_HD, 1 << 0)
    self.assertEqual(OFPPF_10MB_FD, 1 << 1)
    self.assertEqual(OFPPF_100MB_HD, 1 << 2)
    self.assertEqual(OFPPF_100MB_FD, 1 << 3)
    self.assertEqual(OFPPF_1GB_HD, 1 << 4)
    self.assertEqual(OFPPF_1GB_FD, 1 << 5)
    self.assertEqual(OFPPF_10GB_FD, 1 << 6)
    self.assertEqual(OFPPF_40GB_FD, 1 << 7)
    self.assertEqual(OFPPF_100GB_FD, 1 << 8)
    self.assertEqual(OFPPF_1TB_FD, 1 << 9)
    self.assertEqual(OFPPF_OTHER, 1 << 10)
    self.assertEqual(OFPPF_COPPER, 1 << 11)
    self.assertEqual(OFPPF_FIBER, 1 << 12)
    self.assertEqual(OFPPF_AUTONEG, 1 << 13)
    self.assertEqual(OFPPF_PAUSE, 1 << 14)
    self.assertEqual(OFPPF_PAUSE_ASYM, 1 << 15)