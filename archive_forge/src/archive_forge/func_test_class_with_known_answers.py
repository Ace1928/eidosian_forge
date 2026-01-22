import unittest
from array import array
import binascii
from .crcmod import mkCrcFun, Crc
from .crcmod import _usingExtension
from .predefined import PredefinedCrc
from .predefined import mkPredefinedCrcFun
from .predefined import _crc_definitions as _predefined_crc_definitions
def test_class_with_known_answers(self):
    for crcfun_name, v in self.known_answers:
        for i, msg in enumerate(self.test_messages_for_known_answers):
            crc1 = PredefinedCrc(crcfun_name)
            crc1.update(msg)
            self.assertEqual(crc1.crcValue, v[i], "Wrong answer for crc1 %s, input '%s'" % (crcfun_name, msg))
            crc2 = crc1.new()
            self.assertEqual(crc1.crcValue, v[i], "Wrong state for crc1 %s, input '%s'" % (crcfun_name, msg))
            self.assertEqual(crc2.crcValue, v[0], "Wrong state for crc2 %s, input '%s'" % (crcfun_name, msg))
            crc2.update(msg)
            self.assertEqual(crc1.crcValue, v[i], "Wrong state for crc1 %s, input '%s'" % (crcfun_name, msg))
            self.assertEqual(crc2.crcValue, v[i], "Wrong state for crc2 %s, input '%s'" % (crcfun_name, msg))