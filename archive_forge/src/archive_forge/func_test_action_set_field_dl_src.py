import logging
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ether
from os_ken.ofproto import inet
from os_ken.tests.integrated import tester
def test_action_set_field_dl_src(self, dp):
    field = dp.ofproto.OXM_OF_ETH_SRC
    dl_src = '08:82:63:b6:62:05'
    value = self.haddr_to_bin(dl_src)
    self.add_set_field_action(dp, field, value)