import inspect
import logging
import struct
import unittest
from os_ken.lib.packet import icmp
from os_ken.lib.packet import packet_utils
def setUp_with_TimeExceeded(self):
    self.te_data = b'abc'
    self.te_data_len = len(self.te_data)
    self.data = icmp.TimeExceeded(data_len=self.te_data_len, data=self.te_data)
    self.type_ = icmp.ICMP_TIME_EXCEEDED
    self.code = 0
    self.ic = icmp.icmp(self.type_, self.code, self.csum, self.data)
    self.buf = bytearray(struct.pack(icmp.icmp._PACK_STR, self.type_, self.code, self.csum))
    self.buf += self.data.serialize()
    self.csum_calc = packet_utils.checksum(self.buf)
    struct.pack_into('!H', self.buf, 2, self.csum_calc)