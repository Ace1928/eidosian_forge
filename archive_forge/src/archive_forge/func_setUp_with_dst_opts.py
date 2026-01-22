import unittest
import logging
import inspect
import struct
from os_ken.lib import addrconv
from os_ken.lib import ip
from os_ken.lib.packet import ipv6
def setUp_with_dst_opts(self):
    self.opt1_type = 5
    self.opt1_len = 2
    self.opt1_data = b'\x00\x00'
    self.opt2_type = 1
    self.opt2_len = 0
    self.opt2_data = None
    self.options = [ipv6.option(self.opt1_type, self.opt1_len, self.opt1_data), ipv6.option(self.opt2_type, self.opt2_len, self.opt2_data)]
    self.dst_opts_nxt = 6
    self.dst_opts_size = 0
    self.dst_opts = ipv6.dst_opts(self.dst_opts_nxt, self.dst_opts_size, self.options)
    self.ext_hdrs = [self.dst_opts]
    self.payload_length += len(self.dst_opts)
    self.nxt = ipv6.dst_opts.TYPE
    self.ip = ipv6.ipv6(self.version, self.traffic_class, self.flow_label, self.payload_length, self.nxt, self.hop_limit, self.src, self.dst, self.ext_hdrs)
    self.buf = struct.pack(ipv6.ipv6._PACK_STR, self.v_tc_flow, self.payload_length, self.nxt, self.hop_limit, addrconv.ipv6.text_to_bin(self.src), addrconv.ipv6.text_to_bin(self.dst))
    self.buf += self.dst_opts.serialize()