import logging
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ether
from os_ken.ofproto import inet
from os_ken.tests.integrated import tester
def test_action_set_field_icmpv6_type(self, dp):
    field = dp.ofproto.OXM_OF_ICMPV6_TYPE
    value = 129
    self.add_set_field_action(dp, field, value)