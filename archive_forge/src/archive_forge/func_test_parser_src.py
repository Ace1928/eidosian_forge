import unittest
import logging
from os_ken.ofproto.ofproto_v1_0_parser import *
from os_ken.ofproto.nx_actions import *
from os_ken.ofproto import ofproto_v1_0_parser
from os_ken.lib import addrconv
def test_parser_src(self):
    type_ = {'buf': b'\x00\t', 'val': ofproto.OFPAT_SET_TP_SRC}
    buf = type_['buf'] + self.len_['buf'] + self.tp['buf'] + self.zfill
    res = self.c.parser(buf, 0)
    self.assertEqual(self.tp['val'], res.tp)