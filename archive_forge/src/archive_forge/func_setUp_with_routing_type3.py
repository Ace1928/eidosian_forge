import unittest
import logging
import inspect
import struct
from os_ken.lib import addrconv
from os_ken.lib import ip
from os_ken.lib.packet import ipv6
def setUp_with_routing_type3(self):
    self.routing_nxt = 6
    self.routing_size = 6
    self.routing_type = 3
    self.routing_seg = 2
    self.routing_cmpi = 0
    self.routing_cmpe = 0
    self.routing_adrs = ['2001:db8:dead::1', '2001:db8:dead::2', '2001:db8:dead::3']
    self.routing = ipv6.routing_type3(self.routing_nxt, self.routing_size, self.routing_type, self.routing_seg, self.routing_cmpi, self.routing_cmpe, self.routing_adrs)
    self.ext_hdrs = [self.routing]
    self.payload_length += len(self.routing)
    self.nxt = ipv6.routing.TYPE
    self.ip = ipv6.ipv6(self.version, self.traffic_class, self.flow_label, self.payload_length, self.nxt, self.hop_limit, self.src, self.dst, self.ext_hdrs)
    self.buf = struct.pack(ipv6.ipv6._PACK_STR, self.v_tc_flow, self.payload_length, self.nxt, self.hop_limit, addrconv.ipv6.text_to_bin(self.src), addrconv.ipv6.text_to_bin(self.dst))
    self.buf += self.routing.serialize()