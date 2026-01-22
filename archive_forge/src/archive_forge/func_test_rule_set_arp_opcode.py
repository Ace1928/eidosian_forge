import logging
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ether
from os_ken.ofproto import inet
from os_ken.tests.integrated import tester
def test_rule_set_arp_opcode(self, dp):
    dl_type = ether.ETH_TYPE_ARP
    arp_op = 1
    headers = [dp.ofproto.OXM_OF_ARP_OP]
    self._set_verify(headers, arp_op)
    match = dp.ofproto_parser.OFPMatch()
    match.set_dl_type(dl_type)
    match.set_arp_opcode(arp_op)
    self.add_matches(dp, match)