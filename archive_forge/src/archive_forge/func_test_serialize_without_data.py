import unittest
import logging
import struct
import inspect
from os_ken.ofproto import ether, inet
from os_ken.lib.packet.ethernet import ethernet
from os_ken.lib.packet import icmpv6
from os_ken.lib.packet.ipv6 import ipv6
from os_ken.lib.packet import packet_utils
from os_ken.lib import addrconv
def test_serialize_without_data(self):
    rs = icmpv6.nd_router_solicit(self.res)
    prev = ipv6(6, 0, 0, 8, 64, 255, self.src_ipv6, self.dst_ipv6)
    rs_csum = icmpv6_csum(prev, self.buf)
    icmp = icmpv6.icmpv6(self.type_, self.code, 0, rs)
    buf = bytes(icmp.serialize(bytearray(), prev))
    type_, code, csum = struct.unpack_from(icmp._PACK_STR, buf, 0)
    res = struct.unpack_from(rs._PACK_STR, buf, icmp._MIN_LEN)
    data = buf[icmp._MIN_LEN + rs._MIN_LEN:]
    self.assertEqual(type_, self.type_)
    self.assertEqual(code, self.code)
    self.assertEqual(csum, rs_csum)
    self.assertEqual(res[0], self.res)
    self.assertEqual(data, b'')