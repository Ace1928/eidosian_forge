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
def test_serialize_with_cookie_ack(self):
    self.setUp_with_cookie_ack()
    buf = self._test_serialize()
    res = struct.unpack_from(sctp.chunk_cookie_ack._PACK_STR, buf)
    self.assertEqual(sctp.chunk_cookie_ack.chunk_type(), res[0])
    self.assertEqual(self.flags, res[1])
    self.assertEqual(self.length, res[2])