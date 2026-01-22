import time
import logging
from os_ken.controller import ofp_event
from os_ken.controller.handler import MAIN_DISPATCHER
from os_ken.controller.handler import set_ev_cls
from os_ken.ofproto import ofproto_v1_2
from os_ken.tests.integrated import tester
def test_aggregate_stats_flow_count_out_port(self, dp):
    actions = [dp.ofproto_parser.OFPActionOutput(1, 1500)]
    self.mod_flow(dp, table_id=1, actions=actions)
    actions = [dp.ofproto_parser.OFPActionOutput(2, 1500)]
    self.mod_flow(dp, table_id=2, actions=actions)
    dp.send_barrier()
    out_port = 2
    match = dp.ofproto_parser.OFPMatch()
    m = dp.ofproto_parser.OFPAggregateStatsRequest(dp, dp.ofproto.OFPTT_ALL, out_port, dp.ofproto.OFPG_ANY, 0, 0, match)
    dp.send_msg(m)