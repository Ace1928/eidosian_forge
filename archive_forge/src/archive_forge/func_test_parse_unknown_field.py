import unittest
import logging
import socket
from struct import *
from os_ken.ofproto.ofproto_v1_2_parser import *
from os_ken.ofproto import ofproto_v1_2_parser
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ofproto_protocol
from os_ken.ofproto import ether
from os_ken.ofproto.ofproto_parser import MsgBase
from os_ken import utils
from os_ken.lib import addrconv
from os_ken.lib import pack_utils
def test_parse_unknown_field(self):
    buf = bytearray()
    pack_utils.msg_pack_into('!HH', buf, 0, ofproto.OFPMT_OXM, 4 + 6)
    header = ofproto.oxm_tlv_header(36, 2)
    pack_utils.msg_pack_into('!IH', buf, 4, header, 1)
    header = ofproto.OXM_OF_ETH_TYPE
    pack_utils.msg_pack_into('!IH', buf, 10, header, 1)
    match = OFPMatch()
    res = match.parser(bytes(buf), 0)