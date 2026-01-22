import unittest
import logging
from os_ken.ofproto.ofproto_v1_2 import *
def test_enum_ofp_controller_role(self):
    self.assertEqual(OFPCR_ROLE_NOCHANGE, 0)
    self.assertEqual(OFPCR_ROLE_EQUAL, 1)
    self.assertEqual(OFPCR_ROLE_MASTER, 2)
    self.assertEqual(OFPCR_ROLE_SLAVE, 3)