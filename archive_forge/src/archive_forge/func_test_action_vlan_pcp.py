import logging
from os_ken.tests.integrated import tester
from os_ken.ofproto import ofproto_v1_0
from os_ken.ofproto import ether
from os_ken.ofproto import nx_match
def test_action_vlan_pcp(self, dp):
    vlan_pcp = 4
    self._verify = [dp.ofproto.OFPAT_SET_VLAN_PCP, 'vlan_pcp', vlan_pcp]
    action = dp.ofproto_parser.OFPActionVlanPcp(vlan_pcp)
    self.add_action(dp, [action])