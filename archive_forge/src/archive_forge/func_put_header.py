import struct
from os_ken import exception
from os_ken.lib import mac
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.ofproto import ether
from os_ken.ofproto import ofproto_parser
from os_ken.ofproto import ofproto_v1_0
from os_ken.ofproto import inet
import logging
def put_header(self, buf, offset):
    msg_pack_into(ofproto_v1_0.NXM_HEADER_PACK_STRING, buf, offset, self.header)
    return struct.calcsize(ofproto_v1_0.NXM_HEADER_PACK_STRING)