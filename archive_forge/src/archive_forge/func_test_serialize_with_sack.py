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
def test_serialize_with_sack(self):
    self.setUp_with_sack()
    buf = self._test_serialize()
    res = struct.unpack_from(sctp.chunk_sack._PACK_STR, buf)
    self.assertEqual(sctp.chunk_sack.chunk_type(), res[0])
    self.assertEqual(self.flags, res[1])
    self.assertEqual(self.length, res[2])
    self.assertEqual(self.tsn_ack, res[3])
    self.assertEqual(self.a_rwnd, res[4])
    self.assertEqual(self.gapack_num, res[5])
    self.assertEqual(self.duptsn_num, res[6])
    buf = buf[sctp.chunk_sack._MIN_LEN:]
    gapacks = []
    for _ in range(self.gapack_num):
        gap_s, gap_e = struct.unpack_from(sctp.chunk_sack._GAPACK_STR, buf)
        one = [gap_s, gap_e]
        gapacks.append(one)
        buf = buf[sctp.chunk_sack._GAPACK_LEN:]
    duptsns = []
    for _ in range(self.duptsn_num):
        duptsn, = struct.unpack_from(sctp.chunk_sack._DUPTSN_STR, buf)
        duptsns.append(duptsn)
        buf = buf[sctp.chunk_sack._DUPTSN_LEN:]
    self.assertEqual(self.gapacks, gapacks)
    self.assertEqual(self.duptsns, duptsns)