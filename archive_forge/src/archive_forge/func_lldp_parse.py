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
@staticmethod
def lldp_parse(data):
    pkt = packet.Packet(data)
    i = iter(pkt)
    eth_pkt = next(i)
    assert type(eth_pkt) == ethernet.ethernet
    lldp_pkt = next(i)
    if type(lldp_pkt) != lldp.lldp:
        raise LLDPPacket.LLDPUnknownFormat()
    tlv_chassis_id = lldp_pkt.tlvs[0]
    if tlv_chassis_id.subtype != lldp.ChassisID.SUB_LOCALLY_ASSIGNED:
        raise LLDPPacket.LLDPUnknownFormat(msg='unknown chassis id subtype %d' % tlv_chassis_id.subtype)
    chassis_id = tlv_chassis_id.chassis_id.decode('utf-8')
    if not chassis_id.startswith(LLDPPacket.CHASSIS_ID_PREFIX):
        raise LLDPPacket.LLDPUnknownFormat(msg='unknown chassis id format %s' % chassis_id)
    src_dpid = str_to_dpid(chassis_id[LLDPPacket.CHASSIS_ID_PREFIX_LEN:])
    tlv_port_id = lldp_pkt.tlvs[1]
    if tlv_port_id.subtype != lldp.PortID.SUB_PORT_COMPONENT:
        raise LLDPPacket.LLDPUnknownFormat(msg='unknown port id subtype %d' % tlv_port_id.subtype)
    port_id = tlv_port_id.port_id
    if len(port_id) != LLDPPacket.PORT_ID_SIZE:
        raise LLDPPacket.LLDPUnknownFormat(msg='unknown port id %d' % port_id)
    src_port_no, = struct.unpack(LLDPPacket.PORT_ID_STR, port_id)
    return (src_dpid, src_port_no)