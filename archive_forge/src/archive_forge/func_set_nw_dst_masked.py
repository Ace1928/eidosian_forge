import struct
from os_ken import exception
from os_ken.lib import mac
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.ofproto import ether
from os_ken.ofproto import ofproto_parser
from os_ken.ofproto import ofproto_v1_0
from os_ken.ofproto import inet
import logging
def set_nw_dst_masked(self, nw_dst, mask):
    self.flow.nw_dst = nw_dst
    self.wc.nw_dst_mask = mask