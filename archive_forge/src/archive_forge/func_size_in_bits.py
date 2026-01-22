import sys
from Cryptodome.Util.py3compat import tobytes, is_native_int
from Cryptodome.Util._raw_api import (backend, load_lib,
from ._IntegerBase import IntegerBase
def size_in_bits(self):
    """Return the minimum number of bits that can encode the number."""
    if self < 0:
        raise ValueError('Conversion only valid for non-negative numbers')
    return _gmp.mpz_sizeinbase(self._mpz_p, 2)