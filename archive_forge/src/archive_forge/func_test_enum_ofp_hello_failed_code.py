import unittest
import logging
from os_ken.ofproto.ofproto_v1_2 import *
def test_enum_ofp_hello_failed_code(self):
    self.assertEqual(OFPHFC_INCOMPATIBLE, 0)
    self.assertEqual(OFPHFC_EPERM, 1)