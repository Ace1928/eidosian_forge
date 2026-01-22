import unittest
import logging
import inspect
import struct
from os_ken.lib import addrconv
from os_ken.lib import ip
from os_ken.lib.packet import ipv6
def test_serialize_with_adrs_zero(self):
    nxt = 0
    size = 0
    type_ = 3
    seg = 0
    cmpi = 0
    cmpe = 0
    adrs = []
    pad = (8 - ((len(adrs) - 1) * (16 - cmpi) + (16 - cmpe) % 8)) % 8
    routing = ipv6.routing_type3(nxt, size, type_, seg, cmpi, cmpe, pad)
    buf = routing.serialize()
    form = '!BBBBBB2x'
    res = struct.unpack_from(form, bytes(buf))
    self.assertEqual(nxt, res[0])
    self.assertEqual(size, res[1])
    self.assertEqual(type_, res[2])
    self.assertEqual(seg, res[3])
    self.assertEqual(cmpi, res[4] >> 4)
    self.assertEqual(cmpe, res[4] & 15)
    self.assertEqual(pad, res[5])