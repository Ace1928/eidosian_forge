from __future__ import annotations
import abc
import typing
from math import gcd
from cryptography.hazmat.primitives import _serialization, hashes
from cryptography.hazmat.primitives._asymmetric import AsymmetricPadding
from cryptography.hazmat.primitives.asymmetric import utils as asym_utils
def rsa_crt_dmq1(private_exponent: int, q: int) -> int:
    """
    Compute the CRT private_exponent % (q - 1) value from the RSA
    private_exponent (d) and q.
    """
    return private_exponent % (q - 1)