from __future__ import annotations
import abc
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives._asymmetric import (
from cryptography.hazmat.primitives.asymmetric import rsa
@property
def mgf(self) -> MGF:
    return self._mgf