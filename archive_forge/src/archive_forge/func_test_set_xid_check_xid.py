import binascii
import unittest
import struct
from os_ken import exception
from os_ken.ofproto import ofproto_common, ofproto_parser
from os_ken.ofproto import ofproto_v1_0, ofproto_v1_0_parser
import logging
def test_set_xid_check_xid(self):
    xid = 2160492514
    c = ofproto_parser.MsgBase(object)
    c.xid = xid
    self.assertRaises(AssertionError, c.set_xid, xid)