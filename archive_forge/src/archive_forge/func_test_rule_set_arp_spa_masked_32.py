import logging
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ether
from os_ken.ofproto import inet
from os_ken.tests.integrated import tester
def test_rule_set_arp_spa_masked_32(self, dp):
    dl_type = ether.ETH_TYPE_ARP
    nw_src = '192.168.222.57'
    nw_src_int = self.ipv4_to_int(nw_src)
    mask = '255.255.255.255'
    mask_int = self.ipv4_to_int(mask)
    headers = [dp.ofproto.OXM_OF_ARP_SPA, dp.ofproto.OXM_OF_ARP_SPA_W]
    self._set_verify(headers, nw_src_int, mask_int, True, type_='ipv4')
    match = dp.ofproto_parser.OFPMatch()
    match.set_dl_type(dl_type)
    match.set_arp_spa_masked(nw_src_int, mask_int)
    self.add_matches(dp, match)