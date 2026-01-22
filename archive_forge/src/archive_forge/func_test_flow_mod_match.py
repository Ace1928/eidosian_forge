import time
import logging
from os_ken.controller import ofp_event
from os_ken.controller.handler import MAIN_DISPATCHER
from os_ken.controller.handler import set_ev_cls
from os_ken.ofproto import ofproto_v1_2
from os_ken.tests.integrated import tester
def test_flow_mod_match(self, dp):
    self._add_flow_for_flow_mod_tests(dp)
    action = dp.ofproto_parser.OFPActionOutput(3, 1500)
    self._verify[3][3] = action
    match = dp.ofproto_parser.OFPMatch()
    match.set_dl_dst(b'\xff' * 6)
    table_id = 3
    self.mod_flow(dp, command=dp.ofproto.OFPFC_MODIFY, actions=[action], table_id=table_id, match=match)
    dp.send_barrier()
    self.send_flow_stats(dp)