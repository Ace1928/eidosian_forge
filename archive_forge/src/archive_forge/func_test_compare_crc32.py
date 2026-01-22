import unittest
from array import array
import binascii
from .crcmod import mkCrcFun, Crc
from .crcmod import _usingExtension
from .predefined import PredefinedCrc
from .predefined import mkPredefinedCrcFun
from .predefined import _crc_definitions as _predefined_crc_definitions
def test_compare_crc32(self):
    """The binascii module has a 32-bit CRC function that is used in a wide range
        of applications including the checksum used in the ZIP file format.
        This test compares the CRC-32 implementation of this crcmod module to
        that of binascii.crc32."""
    crc32 = mkCrcFun(g32, 0, 1, 4294967295)
    for msg in self.test_messages:
        self.assertEqual(crc32(msg), self.reference_crc32(msg))