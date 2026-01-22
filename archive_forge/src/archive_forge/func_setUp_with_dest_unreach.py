import inspect
import logging
import struct
import unittest
from os_ken.lib.packet import icmp
from os_ken.lib.packet import packet_utils
def setUp_with_dest_unreach(self):
    self.unreach_mtu = 10
    self.unreach_data = b'abc'
    self.unreach_data_len = len(self.unreach_data)
    self.data = icmp.dest_unreach(data_len=self.unreach_data_len, mtu=self.unreach_mtu, data=self.unreach_data)
    self.type_ = icmp.ICMP_DEST_UNREACH
    self.code = icmp.ICMP_HOST_UNREACH_CODE
    self.ic = icmp.icmp(self.type_, self.code, self.csum, self.data)
    self.buf = bytearray(struct.pack(icmp.icmp._PACK_STR, self.type_, self.code, self.csum))
    self.buf += self.data.serialize()
    self.csum_calc = packet_utils.checksum(self.buf)
    struct.pack_into('!H', self.buf, 2, self.csum_calc)