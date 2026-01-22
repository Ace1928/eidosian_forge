import inspect
import logging
import struct
import unittest
from os_ken.lib.packet import icmp
from os_ken.lib.packet import packet_utils
def setUp_with_echo(self):
    self.echo_id = 13379
    self.echo_seq = 1
    self.echo_data = b'0\x0e\t\x00\x00\x00\x00\x00' + b'\x10\x11\x12\x13\x14\x15\x16\x17' + b'\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f' + b' !"#$%&\'' + b'()*+,-./' + b'01234567'
    self.data = icmp.echo(id_=self.echo_id, seq=self.echo_seq, data=self.echo_data)
    self.type_ = icmp.ICMP_ECHO_REQUEST
    self.code = 0
    self.ic = icmp.icmp(self.type_, self.code, self.csum, self.data)
    self.buf = bytearray(struct.pack(icmp.icmp._PACK_STR, self.type_, self.code, self.csum))
    self.buf += self.data.serialize()
    self.csum_calc = packet_utils.checksum(self.buf)
    struct.pack_into('!H', self.buf, 2, self.csum_calc)