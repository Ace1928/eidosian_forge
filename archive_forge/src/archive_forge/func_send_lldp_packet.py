import logging
import struct
import time
from os_ken import cfg
from collections import defaultdict
from os_ken.topology import event
from os_ken.base import app_manager
from os_ken.controller import ofp_event
from os_ken.controller.handler import set_ev_cls
from os_ken.controller.handler import MAIN_DISPATCHER, DEAD_DISPATCHER
from os_ken.exception import OSKenException
from os_ken.lib import addrconv, hub
from os_ken.lib.mac import DONTCARE_STR
from os_ken.lib.dpid import dpid_to_str, str_to_dpid
from os_ken.lib.port_no import port_no_to_str
from os_ken.lib.packet import packet, ethernet
from os_ken.lib.packet import lldp, ether_types
from os_ken.ofproto.ether import ETH_TYPE_LLDP
from os_ken.ofproto.ether import ETH_TYPE_CFM
from os_ken.ofproto import nx_match
from os_ken.ofproto import ofproto_v1_0
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ofproto_v1_3
from os_ken.ofproto import ofproto_v1_4
def send_lldp_packet(self, port):
    try:
        port_data = self.ports.lldp_sent(port)
    except KeyError:
        return
    if port_data.is_down:
        return
    dp = self.dps.get(port.dpid, None)
    if dp is None:
        return
    if dp.ofproto.OFP_VERSION == ofproto_v1_0.OFP_VERSION:
        actions = [dp.ofproto_parser.OFPActionOutput(port.port_no)]
        dp.send_packet_out(actions=actions, data=port_data.lldp_data)
    elif dp.ofproto.OFP_VERSION >= ofproto_v1_2.OFP_VERSION:
        actions = [dp.ofproto_parser.OFPActionOutput(port.port_no)]
        out = dp.ofproto_parser.OFPPacketOut(datapath=dp, in_port=dp.ofproto.OFPP_CONTROLLER, buffer_id=dp.ofproto.OFP_NO_BUFFER, actions=actions, data=port_data.lldp_data)
        dp.send_msg(out)
    else:
        LOG.error('cannot send lldp packet. unsupported version. %x', dp.ofproto.OFP_VERSION)