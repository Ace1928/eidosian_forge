import sys
from typing import Tuple
import nacl.exceptions as exc
from nacl._sodium import ffi, lib
from nacl.exceptions import ensure
def nacl_bindings_pick_scrypt_params(opslimit: int, memlimit: int) -> Tuple[int, int, int]:
    """Python implementation of libsodium's pickparams"""
    if opslimit < 32768:
        opslimit = 32768
    r = 8
    if opslimit < memlimit // 32:
        p = 1
        maxn = opslimit // (4 * r)
        for n_log2 in range(1, 63):
            if 2 ** n_log2 > maxn // 2:
                break
    else:
        maxn = memlimit // (r * 128)
        for n_log2 in range(1, 63):
            if 2 ** n_log2 > maxn // 2:
                break
        maxrp = opslimit // 4 // 2 ** n_log2
        if maxrp > 1073741823:
            maxrp = 1073741823
        p = maxrp // r
    return (n_log2, r, p)