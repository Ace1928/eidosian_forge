import unittest
import logging
from os_ken.ofproto.ofproto_v1_2 import *
def test_enum_ofp_table(self):
    self.assertEqual(OFPTT_MAX, 254)
    self.assertEqual(OFPTT_ALL, 255)