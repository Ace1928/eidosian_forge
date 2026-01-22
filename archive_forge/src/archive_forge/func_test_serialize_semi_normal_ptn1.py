import unittest
import logging
import inspect
import struct
from os_ken.lib import addrconv
from os_ken.lib.packet import cfm
def test_serialize_semi_normal_ptn1(self):
    ins = cfm.sender_id_tlv(chassis_id_subtype=self.chassis_id_subtype, chassis_id=self.chassis_id, ma_domain=self.ma_domain)
    buf = ins.serialize()
    form = '!BHBB1sB2sB'
    res = struct.unpack_from(form, bytes(buf))
    self.assertEqual(self._type, res[0])
    self.assertEqual(7, res[1])
    self.assertEqual(self.chassis_id_length, res[2])
    self.assertEqual(self.chassis_id_subtype, res[3])
    self.assertEqual(self.chassis_id, res[4])
    self.assertEqual(self.ma_domain_length, res[5])
    self.assertEqual(self.ma_domain, res[6])
    self.assertEqual(0, res[7])