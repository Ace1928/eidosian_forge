import unittest
import logging
from os_ken.ofproto.ofproto_v1_0_parser import *
from os_ken.ofproto.nx_actions import *
from os_ken.ofproto import ofproto_v1_0_parser
from os_ken.lib import addrconv
def test_parser_type_src(self):
    type_ = {'buf': b'\x00\x04', 'val': ofproto.OFPAT_SET_DL_SRC}
    buf = type_['buf'] + self.len_['buf'] + self.dl_addr + self.zfill
    res = self.c.parser(buf, 0)
    self.assertEqual(self.dl_addr, res.dl_addr)