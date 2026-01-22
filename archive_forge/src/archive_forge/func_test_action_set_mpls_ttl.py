import logging
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ether
from os_ken.ofproto import inet
from os_ken.tests.integrated import tester
def test_action_set_mpls_ttl(self, dp):
    mpls_ttl = 8
    self._verify = [dp.ofproto.OFPAT_SET_MPLS_TTL, 'mpls_ttl', mpls_ttl]
    actions = [dp.ofproto_parser.OFPActionSetMplsTtl(mpls_ttl)]
    match = dp.ofproto_parser.OFPMatch()
    match.set_dl_type(ether.ETH_TYPE_MPLS)
    self.add_apply_actions(dp, actions, match)