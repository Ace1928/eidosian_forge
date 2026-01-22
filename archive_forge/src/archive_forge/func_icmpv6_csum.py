import unittest
import logging
import struct
import inspect
from os_ken.ofproto import ether, inet
from os_ken.lib.packet.ethernet import ethernet
from os_ken.lib.packet import icmpv6
from os_ken.lib.packet.ipv6 import ipv6
from os_ken.lib.packet import packet_utils
from os_ken.lib import addrconv
def icmpv6_csum(prev, buf):
    ph = struct.pack('!16s16sI3xB', addrconv.ipv6.text_to_bin(prev.src), addrconv.ipv6.text_to_bin(prev.dst), prev.payload_length, prev.nxt)
    h = bytearray(buf)
    struct.pack_into('!H', h, 2, 0)
    return packet_utils.checksum(ph + h)