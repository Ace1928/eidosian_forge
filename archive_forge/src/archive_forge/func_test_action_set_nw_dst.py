import logging
from os_ken.tests.integrated import tester
from os_ken.ofproto import ofproto_v1_0
from os_ken.ofproto import ether
from os_ken.ofproto import nx_match
def test_action_set_nw_dst(self, dp):
    nw_dst = '223.201.206.3'
    nw_dst_int = self.ipv4_to_int(nw_dst)
    self._verify = [dp.ofproto.OFPAT_SET_NW_DST, 'nw_addr', nw_dst_int]
    action = dp.ofproto_parser.OFPActionSetNwDst(nw_dst_int)
    self.add_action(dp, [action])