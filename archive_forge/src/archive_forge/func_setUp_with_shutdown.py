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
def setUp_with_shutdown(self):
    self.flags = 0
    self.length = 8
    self.tsn_ack = 123456
    self.shutdown = sctp.chunk_shutdown(tsn_ack=self.tsn_ack)
    self.chunks = [self.shutdown]
    self.sc = sctp.sctp(self.src_port, self.dst_port, self.vtag, self.csum, self.chunks)
    self.buf += b'\x07\x00\x00\x08\x00\x01\xe2@'