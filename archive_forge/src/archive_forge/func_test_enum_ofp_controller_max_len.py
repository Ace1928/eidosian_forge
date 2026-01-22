import unittest
import logging
from os_ken.ofproto.ofproto_v1_2 import *
def test_enum_ofp_controller_max_len(self):
    self.assertEqual(OFPCML_MAX, 65509)
    self.assertEqual(OFPCML_NO_BUFFER, 65535)