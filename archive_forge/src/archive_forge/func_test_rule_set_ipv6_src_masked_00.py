import logging
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ether
from os_ken.ofproto import inet
from os_ken.tests.integrated import tester
def test_rule_set_ipv6_src_masked_00(self, dp):
    dl_type = ether.ETH_TYPE_IPV6
    ipv6_src = '2001:db8:bd05:1d2:288a:1fc0:1:10ee'
    ipv6_src_int = self.ipv6_to_int(ipv6_src)
    mask = '0:0:0:0:0:0:0:0'
    mask_int = self.ipv6_to_int(mask)
    headers = [dp.ofproto.OXM_OF_IPV6_SRC, dp.ofproto.OXM_OF_IPV6_SRC_W]
    self._set_verify(headers, ipv6_src_int, mask_int, type_='ipv6')
    match = dp.ofproto_parser.OFPMatch()
    match.set_dl_type(dl_type)
    match.set_ipv6_src_masked(ipv6_src_int, mask_int)
    self.add_matches(dp, match)