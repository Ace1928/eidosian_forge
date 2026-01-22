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
def test_serialize_check_miss_send_len(self):
    flags = 41186
    miss_send_len = None
    c = OFPSetConfig(_Datapath, flags, miss_send_len)
    self.assertRaises(AssertionError, c.serialize)