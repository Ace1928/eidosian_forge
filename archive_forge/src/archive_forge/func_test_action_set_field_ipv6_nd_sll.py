import logging
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ether
from os_ken.ofproto import inet
from os_ken.tests.integrated import tester
def test_action_set_field_ipv6_nd_sll(self, dp):
    field = dp.ofproto.OXM_OF_IPV6_ND_SLL
    sll = '54:db:3f:3e:27:19'
    value = self.haddr_to_bin(sll)
    self.add_set_field_action(dp, field, value)