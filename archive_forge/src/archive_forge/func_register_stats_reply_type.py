import struct
import base64
from os_ken.lib import addrconv
from os_ken.lib import mac
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.lib.packet import packet
from os_ken import utils
from os_ken.ofproto.ofproto_parser import StringifyMixin, MsgBase
from os_ken.ofproto import ether
from os_ken.ofproto import ofproto_parser
from os_ken.ofproto import ofproto_v1_2 as ofproto
import logging
@staticmethod
def register_stats_reply_type(type_, body_single_struct=False):

    def _register_stats_reply_type(cls):
        OFPStatsReply._STATS_TYPES[type_] = cls
        cls.cls_body_single_struct = body_single_struct
        return cls
    return _register_stats_reply_type