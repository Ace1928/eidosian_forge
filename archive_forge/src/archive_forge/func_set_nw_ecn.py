import struct
from os_ken import exception
from os_ken.lib import mac
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.ofproto import ether
from os_ken.ofproto import ofproto_parser
from os_ken.ofproto import ofproto_v1_0
from os_ken.ofproto import inet
import logging
def set_nw_ecn(self, nw_ecn):
    self.wc.wildcards &= ~FWW_NW_ECN
    self.flow.nw_tos &= ~IP_ECN_MASK
    self.flow.nw_tos |= nw_ecn & IP_ECN_MASK