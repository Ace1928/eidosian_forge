import logging
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ether
from os_ken.ofproto import inet
from os_ken.tests.integrated import tester
def test_action_set_field_ipv6_nd_target(self, dp):
    field = dp.ofproto.OXM_OF_IPV6_ND_TARGET
    target = '5420:db3f:921b:3e33:2791:98f:dd7f:2e19'
    value = self.ipv6_to_int(target)
    self.add_set_field_action(dp, field, value)