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
def setUp_with_shutdown_complete(self):
    self.tflag = 0
    self.length = 4
    self.shutdown_complete = sctp.chunk_shutdown_complete()
    self.chunks = [self.shutdown_complete]
    self.sc = sctp.sctp(self.src_port, self.dst_port, self.vtag, self.csum, self.chunks)
    self.buf += b'\x0e\x00\x00\x04'