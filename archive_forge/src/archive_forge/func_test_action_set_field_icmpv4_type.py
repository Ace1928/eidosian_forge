import logging
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ether
from os_ken.ofproto import inet
from os_ken.tests.integrated import tester
def test_action_set_field_icmpv4_type(self, dp):
    field = dp.ofproto.OXM_OF_ICMPV4_TYPE
    value = 8
    match = dp.ofproto_parser.OFPMatch()
    match.set_ip_proto(inet.IPPROTO_ICMP)
    self.add_set_field_action(dp, field, value, match)