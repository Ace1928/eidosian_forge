import unittest
import inspect
import logging
from struct import pack, unpack_from, pack_into
from os_ken.ofproto import ether
from os_ken.ofproto import inet
from os_ken.lib.packet.ethernet import ethernet
from os_ken.lib.packet.ipv4 import ipv4
from os_ken.lib.packet.packet import Packet
from os_ken.lib.packet.packet_utils import checksum
from os_ken.lib import addrconv
from os_ken.lib.packet.igmp import igmp
from os_ken.lib.packet.igmp import igmpv3_query
from os_ken.lib.packet.igmp import igmpv3_report
from os_ken.lib.packet.igmp import igmpv3_report_group
from os_ken.lib.packet.igmp import IGMP_TYPE_QUERY
from os_ken.lib.packet.igmp import IGMP_TYPE_REPORT_V3
from os_ken.lib.packet.igmp import MODE_IS_INCLUDE
def test_serialize_with_records(self):
    self.setUp_with_records()
    data = bytearray()
    prev = None
    buf = bytes(self.g.serialize(data, prev))
    res = unpack_from(igmpv3_report._PACK_STR, buf)
    offset = igmpv3_report._MIN_LEN
    rec1 = igmpv3_report_group.parser(buf[offset:])
    offset += len(rec1)
    rec2 = igmpv3_report_group.parser(buf[offset:])
    offset += len(rec2)
    rec3 = igmpv3_report_group.parser(buf[offset:])
    offset += len(rec3)
    rec4 = igmpv3_report_group.parser(buf[offset:])
    self.assertEqual(res[0], self.msgtype)
    self.assertEqual(res[1], checksum(self.buf))
    self.assertEqual(res[2], self.record_num)
    self.assertEqual(repr(rec1), repr(self.record1))
    self.assertEqual(repr(rec2), repr(self.record2))
    self.assertEqual(repr(rec3), repr(self.record3))
    self.assertEqual(repr(rec4), repr(self.record4))