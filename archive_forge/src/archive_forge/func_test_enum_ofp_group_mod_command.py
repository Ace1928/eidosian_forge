import unittest
import logging
from os_ken.ofproto.ofproto_v1_2 import *
def test_enum_ofp_group_mod_command(self):
    self.assertEqual(OFPGC_ADD, 0)
    self.assertEqual(OFPGC_MODIFY, 1)
    self.assertEqual(OFPGC_DELETE, 2)