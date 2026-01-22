import logging
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ether
from os_ken.ofproto import inet
from os_ken.tests.integrated import tester
def test_rule_set_ipv6_flabel_masked_00(self, dp):
    dl_type = ether.ETH_TYPE_IPV6
    ipv6_label = 807812
    mask = 0
    headers = [dp.ofproto.OXM_OF_IPV6_FLABEL, dp.ofproto.OXM_OF_IPV6_FLABEL_W]
    self._set_verify(headers, ipv6_label, mask)
    match = dp.ofproto_parser.OFPMatch()
    match.set_dl_type(dl_type)
    match.set_ipv6_flabel_masked(ipv6_label, mask)
    self.add_matches(dp, match)