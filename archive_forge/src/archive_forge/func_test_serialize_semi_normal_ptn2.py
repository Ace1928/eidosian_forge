import unittest
import logging
import inspect
import struct
from os_ken.lib import addrconv
from os_ken.lib.packet import cfm
def test_serialize_semi_normal_ptn2(self):
    ins = cfm.sender_id_tlv(ma_domain=self.ma_domain, ma=self.ma)
    buf = ins.serialize()
    form = '!BHBB2sB3s'
    res = struct.unpack_from(form, bytes(buf))
    self.assertEqual(self._type, res[0])
    self.assertEqual(8, res[1])
    self.assertEqual(0, res[2])
    self.assertEqual(self.ma_domain_length, res[3])
    self.assertEqual(self.ma_domain, res[4])
    self.assertEqual(self.ma_length, res[5])
    self.assertEqual(self.ma, res[6])