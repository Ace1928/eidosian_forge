import logging
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ether
from os_ken.ofproto import inet
from os_ken.tests.integrated import tester
def test_action_set_field_dl_dst(self, dp):
    field = dp.ofproto.OXM_OF_ETH_DST
    dl_dst = 'e2:7a:09:79:0b:0f'
    value = self.haddr_to_bin(dl_dst)
    self.add_set_field_action(dp, field, value)