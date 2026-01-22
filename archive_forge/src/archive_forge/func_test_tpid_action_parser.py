import logging
import os
import sys
import unittest
from os_ken.utils import binary_str
from os_ken.lib import pcaplib
from os_ken.lib.packet import packet
from os_ken.lib.packet import bgp
from os_ken.lib.packet import afi
from os_ken.lib.packet import safi
def test_tpid_action_parser(self):
    action = bgp.BGPFlowSpecTPIDActionCommunity(actions=bgp.BGPFlowSpecTPIDActionCommunity.TI | bgp.BGPFlowSpecTPIDActionCommunity.TO, tpid_1=5, tpid_2=6)
    binmsg = action.serialize()
    msg, rest = bgp.BGPFlowSpecTPIDActionCommunity.parse(binmsg)
    self.assertEqual(str(action), str(msg))
    self.assertEqual(rest, b'')