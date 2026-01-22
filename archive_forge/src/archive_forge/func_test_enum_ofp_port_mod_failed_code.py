import unittest
import logging
from os_ken.ofproto.ofproto_v1_2 import *
def test_enum_ofp_port_mod_failed_code(self):
    self.assertEqual(OFPPMFC_BAD_PORT, 0)
    self.assertEqual(OFPPMFC_BAD_HW_ADDR, 1)
    self.assertEqual(OFPPMFC_BAD_CONFIG, 2)
    self.assertEqual(OFPPMFC_BAD_ADVERTISE, 3)
    self.assertEqual(OFPPMFC_EPERM, 4)