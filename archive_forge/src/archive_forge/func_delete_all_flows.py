import sys
import logging
import itertools
from os_ken import utils
from os_ken.lib import mac
from os_ken.base import app_manager
from os_ken.controller import ofp_event
from os_ken.controller import handler
from os_ken.controller import dpset
from os_ken.controller.handler import MAIN_DISPATCHER
from os_ken.controller.handler import CONFIG_DISPATCHER
from os_ken.controller.handler import set_ev_cls
from os_ken.ofproto import ofproto_v1_0
from os_ken.ofproto import ofproto_v1_2
def delete_all_flows(self, dp):
    if dp.ofproto == ofproto_v1_0:
        match = dp.ofproto_parser.OFPMatch(dp.ofproto.OFPFW_ALL, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        m = dp.ofproto_parser.OFPFlowMod(dp, match, 0, dp.ofproto.OFPFC_DELETE, 0, 0, 0, 0, dp.ofproto.OFPP_NONE, 0, None)
    elif dp.ofproto == ofproto_v1_2:
        match = dp.ofproto_parser.OFPMatch()
        m = dp.ofproto_parser.OFPFlowMod(dp, 0, 0, dp.ofproto.OFPTT_ALL, dp.ofproto.OFPFC_DELETE, 0, 0, 0, 4294967295, dp.ofproto.OFPP_ANY, dp.ofproto.OFPG_ANY, 0, match, [])
    dp.send_msg(m)