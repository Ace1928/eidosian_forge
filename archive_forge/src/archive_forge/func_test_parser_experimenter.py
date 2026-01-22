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
def test_parser_experimenter(self):
    type_ = 65535
    exp_type = 1
    experimenter = 1
    data = b'Error Experimenter Message.'
    fmt = ofproto.OFP_ERROR_EXPERIMENTER_MSG_PACK_STR
    buf = self.buf + pack(fmt, type_, exp_type, experimenter) + data
    res = OFPErrorMsg.parser(object, self.version, self.msg_type, self.msg_len, self.xid, buf)
    self.assertEqual(res.version, self.version)
    self.assertEqual(res.msg_type, self.msg_type)
    self.assertEqual(res.msg_len, self.msg_len)
    self.assertEqual(res.xid, self.xid)
    self.assertEqual(res.type, type_)
    self.assertEqual(res.exp_type, exp_type)
    self.assertEqual(res.experimenter, experimenter)
    self.assertEqual(res.data, data)