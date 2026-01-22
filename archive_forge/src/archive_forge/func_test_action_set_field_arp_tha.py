import logging
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ether
from os_ken.ofproto import inet
from os_ken.tests.integrated import tester
def test_action_set_field_arp_tha(self, dp):
    field = dp.ofproto.OXM_OF_ARP_THA
    arp_tha = '71:c8:72:2f:47:fd'
    value = self.haddr_to_bin(arp_tha)
    match = dp.ofproto_parser.OFPMatch()
    match.set_dl_type(ether.ETH_TYPE_ARP)
    self.add_set_field_action(dp, field, value, match)