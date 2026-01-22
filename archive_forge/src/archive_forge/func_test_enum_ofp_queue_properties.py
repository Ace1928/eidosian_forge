import unittest
import logging
from os_ken.ofproto.ofproto_v1_2 import *
def test_enum_ofp_queue_properties(self):
    self.assertEqual(OFPQT_MIN_RATE, 1)
    self.assertEqual(OFPQT_MAX_RATE, 2)
    self.assertEqual(OFPQT_EXPERIMENTER, 65535)