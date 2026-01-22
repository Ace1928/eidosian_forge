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
@classmethod
def parser_stats_body(cls, buf, msg_len, offset):
    body_cls = cls.cls_stats_body_cls
    body = []
    while offset < msg_len:
        entry = body_cls.parser(buf, offset)
        body.append(entry)
        offset += entry.length
    if cls.cls_body_single_struct:
        return body[0]
    return body