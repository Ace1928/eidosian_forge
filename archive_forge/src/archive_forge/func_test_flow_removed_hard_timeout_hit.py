import time
import logging
from os_ken.controller import ofp_event
from os_ken.controller.handler import MAIN_DISPATCHER
from os_ken.controller.handler import set_ev_cls
from os_ken.ofproto import ofproto_v1_2
from os_ken.tests.integrated import tester
def test_flow_removed_hard_timeout_hit(self, dp):
    reason = dp.ofproto.OFPRR_HARD_TIMEOUT
    hard_timeout = 5
    in_port = 1
    sleep = 2
    self._add_flow_flow_removed(dp, reason, in_port=in_port, hard_timeout=hard_timeout)
    dp.send_barrier()
    time.sleep(sleep)
    output = dp.ofproto.OFPP_TABLE
    actions = [dp.ofproto_parser.OFPActionOutput(output, 0)]
    m = dp.ofproto_parser.OFPPacketOut(dp, 4294967295, in_port, actions, None)
    dp.send_msg(m)