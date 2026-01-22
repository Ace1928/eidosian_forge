from .err import OperationalError
from functools import partial
import hashlib
def ed25519_password(password, scramble):
    """Sign a random scramble with elliptic curve Ed25519.

    Secret and public key are derived from password.
    """
    if not _nacl_bindings:
        _init_nacl()
    h = hashlib.sha512(password).digest()
    s = _scalar_clamp(h[:32])
    r = hashlib.sha512(h[32:] + scramble).digest()
    r = _nacl_bindings.crypto_core_ed25519_scalar_reduce(r)
    R = _nacl_bindings.crypto_scalarmult_ed25519_base_noclamp(r)
    A = _nacl_bindings.crypto_scalarmult_ed25519_base_noclamp(s)
    k = hashlib.sha512(R + A + scramble).digest()
    k = _nacl_bindings.crypto_core_ed25519_scalar_reduce(k)
    ks = _nacl_bindings.crypto_core_ed25519_scalar_mul(k, s)
    S = _nacl_bindings.crypto_core_ed25519_scalar_add(ks, r)
    return R + S