from __future__ import annotations
import collections
import contextlib
import itertools
import typing
from contextlib import contextmanager
from cryptography import utils, x509
from cryptography.exceptions import UnsupportedAlgorithm, _Reasons
from cryptography.hazmat.backends.openssl import aead
from cryptography.hazmat.backends.openssl.ciphers import _CipherContext
from cryptography.hazmat.backends.openssl.cmac import _CMACContext
from cryptography.hazmat.backends.openssl.ec import (
from cryptography.hazmat.backends.openssl.rsa import (
from cryptography.hazmat.bindings._rust import openssl as rust_openssl
from cryptography.hazmat.bindings.openssl import binding
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives._asymmetric import AsymmetricPadding
from cryptography.hazmat.primitives.asymmetric import (
from cryptography.hazmat.primitives.asymmetric.padding import (
from cryptography.hazmat.primitives.asymmetric.types import (
from cryptography.hazmat.primitives.ciphers import (
from cryptography.hazmat.primitives.ciphers.algorithms import (
from cryptography.hazmat.primitives.ciphers.modes import (
from cryptography.hazmat.primitives.serialization import ssh
from cryptography.hazmat.primitives.serialization.pkcs12 import (
def rsa_padding_supported(self, padding: AsymmetricPadding) -> bool:
    if isinstance(padding, PKCS1v15):
        return True
    elif isinstance(padding, PSS) and isinstance(padding._mgf, MGF1):
        if self._fips_enabled and isinstance(padding._mgf._algorithm, hashes.SHA1):
            return True
        else:
            return self.hash_supported(padding._mgf._algorithm)
    elif isinstance(padding, OAEP) and isinstance(padding._mgf, MGF1):
        return self._oaep_hash_supported(padding._mgf._algorithm) and self._oaep_hash_supported(padding._algorithm)
    else:
        return False