import logging
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ether
from os_ken.ofproto import inet
from os_ken.tests.integrated import tester
def test_action_set_field_ipv4_src(self, dp):
    field = dp.ofproto.OXM_OF_IPV4_SRC
    ipv4_src = '192.168.3.92'
    value = self.ipv4_to_int(ipv4_src)
    match = dp.ofproto_parser.OFPMatch()
    match.set_dl_type(ether.ETH_TYPE_IP)
    self.add_set_field_action(dp, field, value, match)