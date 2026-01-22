import binascii
import unittest
import struct
from os_ken import exception
from os_ken.ofproto import ofproto_common, ofproto_parser
from os_ken.ofproto import ofproto_v1_0, ofproto_v1_0_parser
import logging
def test_check_msg_parser(self):
    version, msg_type, msg_len, xid = ofproto_parser.header(self.bufPacketIn)
    version = 255
    self.assertRaises(exception.OFPUnknownVersion, ofproto_parser.msg, self, version, msg_type, msg_len, xid, self.bufPacketIn)