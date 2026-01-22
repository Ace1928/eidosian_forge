import struct
import base64
from os_ken.lib import addrconv
from os_ken.lib import mac
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.lib.packet import packet
from os_ken import exception
from os_ken import utils
from os_ken.ofproto.ofproto_parser import StringifyMixin, MsgBase
from os_ken.ofproto import ether
from os_ken.ofproto import nicira_ext
from os_ken.ofproto import nx_actions
from os_ken.ofproto import ofproto_parser
from os_ken.ofproto import ofproto_common
from os_ken.ofproto import ofproto_v1_3 as ofproto
import logging
def set_ipv6_dst_masked(self, dst, mask):
    self._wc.ft_set(ofproto.OFPXMT_OFB_IPV6_DST)
    self._wc.ipv6_dst_mask = mask
    self._flow.ipv6_dst = [x & y for x, y in zip(dst, mask)]