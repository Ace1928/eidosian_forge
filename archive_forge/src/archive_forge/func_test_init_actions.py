import unittest
import logging
from os_ken.ofproto.ofproto_v1_0_parser import *
from os_ken.ofproto.nx_actions import *
from os_ken.ofproto import ofproto_v1_0_parser
from os_ken.lib import addrconv
def test_init_actions(self):
    c = self._get_obj(self.actions)
    action = c.actions[0]
    self.assertEqual(self.port, action.port)