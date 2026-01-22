import unittest
import logging
from os_ken.ofproto.ofproto_v1_2 import *
def test_enum_ofp_port_reason(self):
    self.assertEqual(OFPPR_ADD, 0)
    self.assertEqual(OFPPR_DELETE, 1)
    self.assertEqual(OFPPR_MODIFY, 2)