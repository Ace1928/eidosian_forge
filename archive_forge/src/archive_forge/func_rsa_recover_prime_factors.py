from __future__ import annotations
import abc
import typing
from math import gcd
from cryptography.hazmat.primitives import _serialization, hashes
from cryptography.hazmat.primitives._asymmetric import AsymmetricPadding
from cryptography.hazmat.primitives.asymmetric import utils as asym_utils
def rsa_recover_prime_factors(n: int, e: int, d: int) -> typing.Tuple[int, int]:
    """
    Compute factors p and q from the private exponent d. We assume that n has
    no more than two factors. This function is adapted from code in PyCrypto.
    """
    ktot = d * e - 1
    t = ktot
    while t % 2 == 0:
        t = t // 2
    spotted = False
    a = 2
    while not spotted and a < _MAX_RECOVERY_ATTEMPTS:
        k = t
        while k < ktot:
            cand = pow(a, k, n)
            if cand != 1 and cand != n - 1 and (pow(cand, 2, n) == 1):
                p = gcd(cand + 1, n)
                spotted = True
                break
            k *= 2
        a += 2
    if not spotted:
        raise ValueError('Unable to compute factors p and q from exponent d.')
    q, r = divmod(n, p)
    assert r == 0
    p, q = sorted((p, q), reverse=True)
    return (p, q)