import unittest
import logging
from os_ken.ofproto.ofproto_v1_2 import *
def test_struct_ofp_switch_feature(self):
    self.assertEqual(OFP_SWITCH_FEATURES_PACK_STR, '!QIB3xII')
    self.assertEqual(OFP_SWITCH_FEATURES_SIZE, 32)