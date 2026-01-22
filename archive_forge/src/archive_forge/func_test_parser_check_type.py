import unittest
import logging
from os_ken.ofproto.ofproto_v1_0_parser import *
from os_ken.ofproto.nx_actions import *
from os_ken.ofproto import ofproto_v1_0_parser
from os_ken.lib import addrconv
def test_parser_check_type(self):
    type_ = {'buf': b'\x00\n', 'val': 10}
    buf = type_['buf'] + self.len_['buf'] + self.port['buf'] + self.zfill + self.queue_id['buf']
    self.assertRaises(AssertionError, self.c.parser, buf, 0)