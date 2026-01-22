import unittest
import logging
import inspect
import struct
from os_ken.lib import addrconv
from os_ken.lib.packet import cfm
def test_serialize_with_md_name_length_zero(self):
    ins = cfm.cc_message(self.md_lv, self.version, self.rdi, self.interval, self.seq_num, self.mep_id, self.md_name_format, 0, self.md_name, self.short_ma_name_format, 0, self.short_ma_name, self.tlvs)
    buf = ins.serialize()
    res = struct.unpack_from(self.form, bytes(buf))
    self.assertEqual(self.md_lv, res[0] >> 5)
    self.assertEqual(self.version, res[0] & 31)
    self.assertEqual(self.opcode, res[1])
    self.assertEqual(self.rdi, res[2] >> 7)
    self.assertEqual(self.interval, res[2] & 7)
    self.assertEqual(self.first_tlv_offset, res[3])
    self.assertEqual(self.seq_num, res[4])
    self.assertEqual(self.mep_id, res[5])
    self.assertEqual(self.md_name_format, res[6])
    self.assertEqual(self.md_name_length, res[7])
    self.assertEqual(self.md_name, res[8])
    self.assertEqual(self.short_ma_name_format, res[9])
    self.assertEqual(self.short_ma_name_length, res[10])
    self.assertEqual(self.short_ma_name, res[11])
    self.assertEqual(self.end_tlv, res[12])