import struct
import socket
import logging
from os_ken.ofproto import ofproto_v1_0
from os_ken.lib import ofctl_utils
from os_ken.lib.mac import haddr_to_bin, haddr_to_str
def delete_flow_entry(dp):
    match = dp.ofproto_parser.OFPMatch(dp.ofproto.OFPFW_ALL, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    flow_mod = dp.ofproto_parser.OFPFlowMod(datapath=dp, match=match, cookie=0, command=dp.ofproto.OFPFC_DELETE)
    ofctl_utils.send_msg(dp, flow_mod, LOG)