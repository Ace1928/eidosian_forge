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
def setUp_with_error(self):
    self.flags = 0
    self.length = 4 + 8 + 16 + 8 + 4 + 20 + 8 + 4 + 8 + 8 + 4 + 12 + 20 + 20
    self.c_invalid_stream_id = sctp.cause_invalid_stream_id(4096)
    self.c_missing_param = sctp.cause_missing_param([sctp.PTYPE_IPV4, sctp.PTYPE_IPV6, sctp.PTYPE_COOKIE_PRESERVE, sctp.PTYPE_HOST_ADDR])
    self.c_stale_cookie = sctp.cause_stale_cookie(b'\x00\x00\x13\x88')
    self.c_out_of_resource = sctp.cause_out_of_resource()
    self.c_unresolvable_addr = sctp.cause_unresolvable_addr(sctp.param_host_addr(b'test host\x00'))
    self.c_unrecognized_chunk = sctp.cause_unrecognized_chunk(b'\xff\x00\x00\x04')
    self.c_invalid_param = sctp.cause_invalid_param()
    self.c_unrecognized_param = sctp.cause_unrecognized_param(b'\xff\xff\x00\x04')
    self.c_no_userdata = sctp.cause_no_userdata(b'\x00\x01\xe2@')
    self.c_cookie_while_shutdown = sctp.cause_cookie_while_shutdown()
    self.c_restart_with_new_addr = sctp.cause_restart_with_new_addr(sctp.param_ipv4('192.168.1.1'))
    self.c_user_initiated_abort = sctp.cause_user_initiated_abort(b'Key Interrupt.\x00')
    self.c_protocol_violation = sctp.cause_protocol_violation(b'Unknown reason.\x00')
    self.causes = [self.c_invalid_stream_id, self.c_missing_param, self.c_stale_cookie, self.c_out_of_resource, self.c_unresolvable_addr, self.c_unrecognized_chunk, self.c_invalid_param, self.c_unrecognized_param, self.c_no_userdata, self.c_cookie_while_shutdown, self.c_restart_with_new_addr, self.c_user_initiated_abort, self.c_protocol_violation]
    self.error = sctp.chunk_error(causes=self.causes)
    self.chunks = [self.error]
    self.sc = sctp.sctp(self.src_port, self.dst_port, self.vtag, self.csum, self.chunks)
    self.buf += b'\t\x00\x00\x90' + b'\x00\x01\x00\x08\x10\x00\x00\x00' + b'\x00\x02\x00\x10\x00\x00\x00\x04' + b'\x00\x05\x00\x06\x00\t\x00\x0b' + b'\x00\x03\x00\x08\x00\x00\x13\x88' + b'\x00\x04\x00\x04' + b'\x00\x05\x00\x14' + b'\x00\x0b\x00\x0e' + b'test host\x00\x00\x00' + b'\x00\x06\x00\x08\xff\x00\x00\x04' + b'\x00\x07\x00\x04' + b'\x00\x08\x00\x08\xff\xff\x00\x04' + b'\x00\t\x00\x08\x00\x01\xe2@' + b'\x00\n\x00\x04' + b'\x00\x0b\x00\x0c' + b'\x00\x05\x00\x08\xc0\xa8\x01\x01' + b'\x00\x0c\x00\x13' + b'Key Inte' + b'rrupt.\x00\x00' + b'\x00\r\x00\x14' + b'Unknown ' + b'reason.\x00'