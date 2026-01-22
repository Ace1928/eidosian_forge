import sys
from Cryptodome.Util.py3compat import tobytes, is_native_int
from Cryptodome.Util._raw_api import (backend, load_lib,
from ._IntegerBase import IntegerBase
def get_bit(self, n):
    """Return True if the n-th bit is set to 1.
        Bit 0 is the least significant."""
    if self < 0:
        raise ValueError('no bit representation for negative values')
    if n < 0:
        raise ValueError('negative bit count')
    if n > 65536:
        return 0
    return bool(_gmp.mpz_tstbit(self._mpz_p, c_ulong(int(n))))