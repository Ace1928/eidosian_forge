import logging
import netaddr
from os_ken.ofproto import ether
from os_ken.ofproto import inet
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ofproto_v1_2_parser
from os_ken.lib import ofctl_utils
def get_aggregate_flow_stats(dp, waiters, flow=None):
    flow = flow if flow else {}
    table_id = UTIL.ofp_table_from_user(flow.get('table_id', dp.ofproto.OFPTT_ALL))
    out_port = UTIL.ofp_port_from_user(flow.get('out_port', dp.ofproto.OFPP_ANY))
    out_group = UTIL.ofp_group_from_user(flow.get('out_group', dp.ofproto.OFPG_ANY))
    cookie = str_to_int(flow.get('cookie', 0))
    cookie_mask = str_to_int(flow.get('cookie_mask', 0))
    match = to_match(dp, flow.get('match', {}))
    stats = dp.ofproto_parser.OFPAggregateStatsRequest(dp, table_id, out_port, out_group, cookie, cookie_mask, match)
    msgs = []
    ofctl_utils.send_stats_request(dp, stats, waiters, msgs, LOG)
    flows = []
    for msg in msgs:
        stats = msg.body
        s = {'packet_count': stats.packet_count, 'byte_count': stats.byte_count, 'flow_count': stats.flow_count}
        flows.append(s)
    return {str(dp.id): flows}