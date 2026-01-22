import time
import logging
from os_ken.controller import ofp_event
from os_ken.controller.handler import MAIN_DISPATCHER
from os_ken.controller.handler import set_ev_cls
from os_ken.ofproto import ofproto_v1_2
from os_ken.tests.integrated import tester
def test_aggregate_stats_flow_count(self, dp):
    c = 0
    while c < self.n_tables:
        self.mod_flow(dp, table_id=c)
        c += 1
    dp.send_barrier()
    match = dp.ofproto_parser.OFPMatch()
    m = dp.ofproto_parser.OFPAggregateStatsRequest(dp, dp.ofproto.OFPTT_ALL, dp.ofproto.OFPP_ANY, dp.ofproto.OFPG_ANY, 0, 0, match)
    dp.send_msg(m)