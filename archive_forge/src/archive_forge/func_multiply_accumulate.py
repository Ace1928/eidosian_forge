import sys
from Cryptodome.Util.py3compat import tobytes, is_native_int
from Cryptodome.Util._raw_api import (backend, load_lib,
from ._IntegerBase import IntegerBase
def multiply_accumulate(self, a, b):
    """Increment the number by the product of a and b."""
    if not isinstance(a, IntegerGMP):
        a = IntegerGMP(a)
    if is_native_int(b):
        if 0 < b < 65536:
            _gmp.mpz_addmul_ui(self._mpz_p, a._mpz_p, c_ulong(b))
            return self
        if -65535 < b < 0:
            _gmp.mpz_submul_ui(self._mpz_p, a._mpz_p, c_ulong(-b))
            return self
        b = IntegerGMP(b)
    _gmp.mpz_addmul(self._mpz_p, a._mpz_p, b._mpz_p)
    return self