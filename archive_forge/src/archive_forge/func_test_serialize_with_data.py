import inspect
import logging
import struct
import unittest
from os_ken.lib import addrconv
from os_ken.lib.packet import packet
from os_ken.lib.packet import ethernet
from os_ken.lib.packet import ipv4
from os_ken.lib.packet import sctp
from os_ken.ofproto import ether
from os_ken.ofproto import inet
def test_serialize_with_data(self):
    self.setUp_with_data()
    buf = self._test_serialize()
    res = struct.unpack_from(sctp.chunk_data._PACK_STR, buf)
    self.assertEqual(sctp.chunk_data.chunk_type(), res[0])
    flags = self.unordered << 2 | self.begin << 1 | self.end << 0
    self.assertEqual(flags, res[1])
    self.assertEqual(self.length, res[2])
    self.assertEqual(self.tsn, res[3])
    self.assertEqual(self.sid, res[4])
    self.assertEqual(self.seq, res[5])
    self.assertEqual(self.payload_id, res[6])
    self.assertEqual(self.payload_data, buf[sctp.chunk_data._MIN_LEN:])