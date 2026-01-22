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
def test_vlan_action_parser(self):
    action = bgp.BGPFlowSpecVlanActionCommunity(actions_1=bgp.BGPFlowSpecVlanActionCommunity.POP | bgp.BGPFlowSpecVlanActionCommunity.SWAP, vlan_1=3000, cos_1=3, actions_2=bgp.BGPFlowSpecVlanActionCommunity.PUSH, vlan_2=4000, cos_2=2)
    binmsg = action.serialize()
    msg, rest = bgp.BGPFlowSpecVlanActionCommunity.parse(binmsg)
    self.assertEqual(str(action), str(msg))
    self.assertEqual(rest, b'')