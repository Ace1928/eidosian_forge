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
@staticmethod
def register_stats_type(body_single_struct=False):

    def _register_stats_type(cls):
        assert cls.cls_stats_type is not None
        assert cls.cls_stats_type not in OFPMultipartReply._STATS_MSG_TYPES
        assert cls.cls_stats_body_cls is not None
        cls.cls_body_single_struct = body_single_struct
        OFPMultipartReply._STATS_MSG_TYPES[cls.cls_stats_type] = cls
        return cls
    return _register_stats_type