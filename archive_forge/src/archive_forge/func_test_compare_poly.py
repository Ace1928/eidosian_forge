import unittest
from array import array
import binascii
from .crcmod import mkCrcFun, Crc
from .crcmod import _usingExtension
from .predefined import PredefinedCrc
from .predefined import mkPredefinedCrcFun
from .predefined import _crc_definitions as _predefined_crc_definitions
def test_compare_poly(self):
    """Compare various CRCs of this crcmod module to a pure
        polynomial-based implementation."""
    for crcfun_params, crc_poly_fun in self.test_poly_crcs:
        crcfun = mkCrcFun(*crcfun_params)
        for msg in self.test_messages:
            self.assertEqual(crcfun(msg), crc_poly_fun(msg))