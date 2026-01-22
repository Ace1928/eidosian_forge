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
def setUp_with_heartbeat(self):
    self.flags = 0
    self.length = 4 + 8
    self.p_heartbeat = sctp.param_heartbeat(b'\x01\x02\x03\x04')
    self.heartbeat = sctp.chunk_heartbeat(info=self.p_heartbeat)
    self.chunks = [self.heartbeat]
    self.sc = sctp.sctp(self.src_port, self.dst_port, self.vtag, self.csum, self.chunks)
    self.buf += b'\x04\x00\x00\x0c' + b'\x00\x01\x00\x08' + b'\x01\x02\x03\x04'