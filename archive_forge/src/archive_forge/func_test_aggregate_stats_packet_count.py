import time
import logging
from os_ken.controller import ofp_event
from os_ken.controller.handler import MAIN_DISPATCHER
from os_ken.controller.handler import set_ev_cls
from os_ken.ofproto import ofproto_v1_2
from os_ken.tests.integrated import tester
def test_aggregate_stats_packet_count(self, dp):
    in_port = 1
    data = 'test'
    self._verify = {'packet_count': 1, 'byte_count': len(data)}
    match = dp.ofproto_parser.OFPMatch()
    match.set_in_port(in_port)
    self.mod_flow(dp, table_id=0, match=match)
    output = dp.ofproto.OFPP_TABLE
    actions = [dp.ofproto_parser.OFPActionOutput(output, 0)]
    m = dp.ofproto_parser.OFPPacketOut(dp, 4294967295, in_port, actions, data)
    dp.send_msg(m)
    dp.send_barrier()
    match = dp.ofproto_parser.OFPMatch()
    m = dp.ofproto_parser.OFPAggregateStatsRequest(dp, dp.ofproto.OFPTT_ALL, dp.ofproto.OFPP_ANY, dp.ofproto.OFPG_ANY, 0, 0, match)
    dp.send_msg(m)