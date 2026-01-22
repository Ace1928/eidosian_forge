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
def setUp_with_init(self):
    self.flags = 0
    self.length = 20 + 8 + 20 + 8 + 4 + 16 + 16
    self.init_tag = 123456
    self.a_rwnd = 9876
    self.os = 3
    self.mis = 3
    self.i_tsn = 123456
    self.p_ipv4 = sctp.param_ipv4('192.168.1.1')
    self.p_ipv6 = sctp.param_ipv6('fe80::647e:1aff:fec4:8284')
    self.p_cookie_preserve = sctp.param_cookie_preserve(5000)
    self.p_ecn = sctp.param_ecn()
    self.p_host_addr = sctp.param_host_addr(b'test host\x00')
    self.p_support_type = sctp.param_supported_addr([sctp.PTYPE_IPV4, sctp.PTYPE_IPV6, sctp.PTYPE_COOKIE_PRESERVE, sctp.PTYPE_ECN, sctp.PTYPE_HOST_ADDR])
    self.params = [self.p_ipv4, self.p_ipv6, self.p_cookie_preserve, self.p_ecn, self.p_host_addr, self.p_support_type]
    self.init = sctp.chunk_init(init_tag=self.init_tag, a_rwnd=self.a_rwnd, os=self.os, mis=self.mis, i_tsn=self.i_tsn, params=self.params)
    self.chunks = [self.init]
    self.sc = sctp.sctp(self.src_port, self.dst_port, self.vtag, self.csum, self.chunks)
    self.buf += b'\x01\x00\x00\\\x00\x01\xe2@\x00\x00&\x94' + b'\x00\x03\x00\x03\x00\x01\xe2@' + b'\x00\x05\x00\x08\xc0\xa8\x01\x01' + b'\x00\x06\x00\x14' + b'\xfe\x80\x00\x00\x00\x00\x00\x00' + b'd~\x1a\xff\xfe\xc4\x82\x84' + b'\x00\t\x00\x08\x00\x00\x13\x88' + b'\x80\x00\x00\x04' + b'\x00\x0b\x00\x0e' + b'test host\x00\x00\x00' + b'\x00\x0c\x00\x0e\x00\x05\x00\x06\x00\t\x80\x00' + b'\x00\x0b\x00\x00'