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
def setUp_with_data(self):
    self.unordered = 1
    self.begin = 1
    self.end = 1
    self.length = 16 + 10
    self.tsn = 12345
    self.sid = 1
    self.seq = 0
    self.payload_id = 0
    self.payload_data = b'\x01\x02\x03\x04\x05\x06\x07\x08\t\n'
    self.data = sctp.chunk_data(unordered=self.unordered, begin=self.begin, end=self.end, tsn=self.tsn, sid=self.sid, payload_data=self.payload_data)
    self.chunks = [self.data]
    self.sc = sctp.sctp(self.src_port, self.dst_port, self.vtag, self.csum, self.chunks)
    self.buf += b'\x00\x07\x00\x1a\x00\x0009\x00\x01\x00\x00' + b'\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n'