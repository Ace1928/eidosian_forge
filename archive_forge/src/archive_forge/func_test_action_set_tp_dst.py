import logging
from os_ken.tests.integrated import tester
from os_ken.ofproto import ofproto_v1_0
from os_ken.ofproto import ether
from os_ken.ofproto import nx_match
def test_action_set_tp_dst(self, dp):
    tp_dst = 15430
    self._verify = [dp.ofproto.OFPAT_SET_TP_DST, 'tp', tp_dst]
    action = dp.ofproto_parser.OFPActionSetTpDst(tp_dst)
    self.add_action(dp, [action])