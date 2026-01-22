import logging
from os_ken.tests.integrated import tester
from os_ken.ofproto import ofproto_v1_0
from os_ken.ofproto import ether
from os_ken.ofproto import nx_match
def test_action_set_tp_src(self, dp):
    tp_src = 55420
    self._verify = [dp.ofproto.OFPAT_SET_TP_SRC, 'tp', tp_src]
    action = dp.ofproto_parser.OFPActionSetTpSrc(tp_src)
    self.add_action(dp, [action])