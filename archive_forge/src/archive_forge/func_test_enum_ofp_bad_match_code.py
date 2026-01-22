import unittest
import logging
from os_ken.ofproto.ofproto_v1_2 import *
def test_enum_ofp_bad_match_code(self):
    self.assertEqual(OFPBMC_BAD_TYPE, 0)
    self.assertEqual(OFPBMC_BAD_LEN, 1)
    self.assertEqual(OFPBMC_BAD_TAG, 2)
    self.assertEqual(OFPBMC_BAD_DL_ADDR_MASK, 3)
    self.assertEqual(OFPBMC_BAD_NW_ADDR_MASK, 4)
    self.assertEqual(OFPBMC_BAD_WILDCARDS, 5)
    self.assertEqual(OFPBMC_BAD_FIELD, 6)
    self.assertEqual(OFPBMC_BAD_VALUE, 7)
    self.assertEqual(OFPBMC_BAD_MASK, 8)
    self.assertEqual(OFPBMC_BAD_PREREQ, 9)
    self.assertEqual(OFPBMC_DUP_FIELD, 10)
    self.assertEqual(OFPBMC_EPERM, 11)