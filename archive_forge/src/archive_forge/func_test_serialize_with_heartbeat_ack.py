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
def test_serialize_with_heartbeat_ack(self):
    self.setUp_with_heartbeat_ack()
    buf = self._test_serialize()
    res = struct.unpack_from(sctp.chunk_heartbeat_ack._PACK_STR, buf)
    self.assertEqual(sctp.chunk_heartbeat_ack.chunk_type(), res[0])
    self.assertEqual(self.flags, res[1])
    self.assertEqual(self.length, res[2])
    buf = buf[sctp.chunk_heartbeat_ack._MIN_LEN:]
    res1 = struct.unpack_from(sctp.param_heartbeat._PACK_STR, buf)
    self.assertEqual(sctp.param_heartbeat.param_type(), res1[0])
    self.assertEqual(12, res1[1])
    self.assertEqual(b'\xff\xee\xdd\xcc\xbb\xaa\x99\x88', buf[sctp.param_heartbeat._MIN_LEN:sctp.param_heartbeat._MIN_LEN + 8])