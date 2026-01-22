import unittest
import logging
import inspect
import struct
from os_ken.lib import addrconv
from os_ken.lib.packet import cfm
def test_parser_with_no_maintenance_domain_name_present(self):
    form = '!4BIH3B8s37x12x4xB'
    buf = struct.pack(form, self.md_lv << 5 | self.version, self.opcode, self.rdi << 7 | self.interval, self.first_tlv_offset, self.seq_num, self.mep_id, cfm.cc_message._MD_FMT_NO_MD_NAME_PRESENT, self.short_ma_name_format, self.short_ma_name_length, self.short_ma_name, self.end_tlv)
    _res = cfm.cc_message.parser(buf)
    if type(_res) is tuple:
        res = _res[0]
    else:
        res = _res
    self.assertEqual(self.md_lv, res.md_lv)
    self.assertEqual(self.version, res.version)
    self.assertEqual(self.rdi, res.rdi)
    self.assertEqual(self.interval, res.interval)
    self.assertEqual(self.seq_num, res.seq_num)
    self.assertEqual(self.mep_id, res.mep_id)
    self.assertEqual(cfm.cc_message._MD_FMT_NO_MD_NAME_PRESENT, res.md_name_format)
    self.assertEqual(self.short_ma_name_format, res.short_ma_name_format)
    self.assertEqual(self.short_ma_name_length, res.short_ma_name_length)
    self.assertEqual(self.short_ma_name, res.short_ma_name)
    self.assertEqual(self.tlvs, res.tlvs)