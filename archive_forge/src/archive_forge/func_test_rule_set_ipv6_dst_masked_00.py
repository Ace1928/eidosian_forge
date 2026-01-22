import logging
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ether
from os_ken.ofproto import inet
from os_ken.tests.integrated import tester
def test_rule_set_ipv6_dst_masked_00(self, dp):
    dl_type = ether.ETH_TYPE_IPV6
    ipv6_dst = 'e9e8:9ea5:7d67:82cc:ca54:1fc0:2d24:f038'
    ipv6_dst_int = self.ipv6_to_int(ipv6_dst)
    mask = '0:0:0:0:0:0:0:0'
    mask_int = self.ipv6_to_int(mask)
    headers = [dp.ofproto.OXM_OF_IPV6_DST, dp.ofproto.OXM_OF_IPV6_DST_W]
    self._set_verify(headers, ipv6_dst_int, mask_int, type_='ipv6')
    match = dp.ofproto_parser.OFPMatch()
    match.set_dl_type(dl_type)
    match.set_ipv6_dst_masked(ipv6_dst_int, mask_int)
    self.add_matches(dp, match)