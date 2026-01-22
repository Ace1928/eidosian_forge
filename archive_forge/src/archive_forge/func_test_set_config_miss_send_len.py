import time
import logging
from os_ken.controller import ofp_event
from os_ken.controller.handler import MAIN_DISPATCHER
from os_ken.controller.handler import set_ev_cls
from os_ken.ofproto import ofproto_v1_2
from os_ken.tests.integrated import tester
def test_set_config_miss_send_len(self, dp):
    flags = dp.ofproto.OFPC_FRAG_NORMAL
    ms_len = 256
    self._verify = ms_len
    m = dp.ofproto_parser.OFPSetConfig(dp, flags, ms_len)
    dp.send_msg(m)
    dp.send_barrier()
    m = dp.ofproto_parser.OFPGetConfigRequest(dp)
    dp.send_msg(m)