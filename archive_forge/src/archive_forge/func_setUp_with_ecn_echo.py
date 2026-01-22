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
def setUp_with_ecn_echo(self):
    self.flags = 0
    self.length = 8
    self.low_tsn = 123456
    self.ecn_echo = sctp.chunk_ecn_echo(low_tsn=self.low_tsn)
    self.chunks = [self.ecn_echo]
    self.sc = sctp.sctp(self.src_port, self.dst_port, self.vtag, self.csum, self.chunks)
    self.buf += b'\x0c\x00\x00\x08\x00\x01\xe2@'