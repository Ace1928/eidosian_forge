import logging
import unittest
from os_ken.lib.packet.bgp import (
from os_ken.services.protocols.bgp.utils.bgp import create_v4flowspec_actions
from os_ken.services.protocols.bgp.utils.bgp import create_v6flowspec_actions
from os_ken.services.protocols.bgp.utils.bgp import create_l2vpnflowspec_actions
def test_create_l2vpnflowspec_actions_all_actions(self):
    actions = {'traffic_rate': {'as_number': 0, 'rate_info': 100.0}, 'traffic_action': {'action': 3}, 'redirect': {'as_number': 10, 'local_administrator': 10}, 'traffic_marking': {'dscp': 24}, 'vlan_action': {'actions_1': BGPFlowSpecVlanActionCommunity.POP | BGPFlowSpecVlanActionCommunity.SWAP, 'vlan_1': 3000, 'cos_1': 3, 'actions_2': BGPFlowSpecVlanActionCommunity.PUSH, 'vlan_2': 4000, 'cos_2': 2}, 'tpid_action': {'actions': BGPFlowSpecTPIDActionCommunity.TI | BGPFlowSpecTPIDActionCommunity.TO, 'tpid_1': 5, 'tpid_2': 6}}
    expected_communities = [BGPFlowSpecTrafficRateCommunity(as_number=0, rate_info=100.0), BGPFlowSpecTrafficActionCommunity(action=3), BGPFlowSpecRedirectCommunity(as_number=10, local_administrator=10), BGPFlowSpecTrafficMarkingCommunity(dscp=24), BGPFlowSpecVlanActionCommunity(actions_1=BGPFlowSpecVlanActionCommunity.POP | BGPFlowSpecVlanActionCommunity.SWAP, vlan_1=3000, cos_1=3, actions_2=BGPFlowSpecVlanActionCommunity.PUSH, vlan_2=4000, cos_2=2), BGPFlowSpecTPIDActionCommunity(actions=BGPFlowSpecTPIDActionCommunity.TI | BGPFlowSpecTPIDActionCommunity.TO, tpid_1=5, tpid_2=6)]
    self._test_create_l2vpnflowspec_actions(actions, expected_communities)