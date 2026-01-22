from Cryptodome.Signature.pss import MGF1
import Cryptodome.Hash.SHA1
from Cryptodome.Util.py3compat import _copy_bytes
import Cryptodome.Util.number
from Cryptodome.Util.number import ceil_div, bytes_to_long, long_to_bytes
from Cryptodome.Util.strxor import strxor
from Cryptodome import Random
from ._pkcs1_oaep_decode import oaep_decode
def can_decrypt(self):
    """Legacy function to check if you can call :meth:`decrypt`.

        .. deprecated:: 3.0"""
    return self._key.can_decrypt()