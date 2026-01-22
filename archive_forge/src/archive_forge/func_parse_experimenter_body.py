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
def parse_experimenter_body(cls, buf):
    type_, exp_type, experimenter = struct.unpack_from(ofproto.OFP_ERROR_EXPERIMENTER_MSG_PACK_STR, buf, ofproto.OFP_HEADER_SIZE)
    data = buf[ofproto.OFP_ERROR_EXPERIMENTER_MSG_SIZE:]
    return (type_, exp_type, experimenter, data)