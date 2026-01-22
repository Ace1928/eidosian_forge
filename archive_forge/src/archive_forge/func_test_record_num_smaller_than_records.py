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
def test_record_num_smaller_than_records(self):
    self.record1 = igmpv3_report_group(MODE_IS_INCLUDE, 0, 0, '225.0.0.1')
    self.record2 = igmpv3_report_group(MODE_IS_INCLUDE, 0, 2, '225.0.0.2', ['172.16.10.10', '172.16.10.27'])
    self.record3 = igmpv3_report_group(MODE_IS_INCLUDE, 1, 0, '225.0.0.3', [], b'abc\x00')
    self.record4 = igmpv3_report_group(MODE_IS_INCLUDE, 1, 2, '225.0.0.4', ['172.16.10.10', '172.16.10.27'], b'abc\x00')
    self.records = [self.record1, self.record2, self.record3, self.record4]
    self.record_num = len(self.records) - 1
    self.buf = pack(igmpv3_report._PACK_STR, self.msgtype, self.csum, self.record_num)
    self.buf += self.record1.serialize()
    self.buf += self.record2.serialize()
    self.buf += self.record3.serialize()
    self.buf += self.record4.serialize()
    self.g = igmpv3_report(self.msgtype, self.csum, self.record_num, self.records)
    self.assertRaises(Exception, self.test_parser)