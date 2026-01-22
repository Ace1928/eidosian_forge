import logging
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ether
from os_ken.ofproto import inet
from os_ken.tests.integrated import tester
def test_rule_set_dl_dst_masked_ff(self, dp):
    dl_dst = 'd0:98:79:b4:75:b5'
    dl_dst_bin = self.haddr_to_bin(dl_dst)
    mask = 'ff:ff:ff:ff:ff:ff'
    mask_bin = self.haddr_to_bin(mask)
    headers = [dp.ofproto.OXM_OF_ETH_DST, dp.ofproto.OXM_OF_ETH_DST_W]
    self._set_verify(headers, dl_dst_bin, mask_bin, True, type_='mac')
    match = dp.ofproto_parser.OFPMatch()
    match.set_dl_dst_masked(dl_dst_bin, mask_bin)
    self.add_matches(dp, match)