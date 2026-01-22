import unittest
import logging
import inspect
import struct
from os_ken.lib import addrconv
from os_ken.lib import ip
from os_ken.lib.packet import ipv6
def setUp_with_fragment(self):
    self.fragment_nxt = 6
    self.fragment_offset = 50
    self.fragment_more = 1
    self.fragment_id = 123
    self.fragment = ipv6.fragment(self.fragment_nxt, self.fragment_offset, self.fragment_more, self.fragment_id)
    self.ext_hdrs = [self.fragment]
    self.payload_length += len(self.fragment)
    self.nxt = ipv6.fragment.TYPE
    self.ip = ipv6.ipv6(self.version, self.traffic_class, self.flow_label, self.payload_length, self.nxt, self.hop_limit, self.src, self.dst, self.ext_hdrs)
    self.buf = struct.pack(ipv6.ipv6._PACK_STR, self.v_tc_flow, self.payload_length, self.nxt, self.hop_limit, addrconv.ipv6.text_to_bin(self.src), addrconv.ipv6.text_to_bin(self.dst))
    self.buf += self.fragment.serialize()