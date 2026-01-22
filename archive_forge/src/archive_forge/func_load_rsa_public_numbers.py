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
def load_rsa_public_numbers(self, numbers: rsa.RSAPublicNumbers) -> rsa.RSAPublicKey:
    rsa._check_public_key_components(numbers.e, numbers.n)
    rsa_cdata = self._lib.RSA_new()
    self.openssl_assert(rsa_cdata != self._ffi.NULL)
    rsa_cdata = self._ffi.gc(rsa_cdata, self._lib.RSA_free)
    e = self._int_to_bn(numbers.e)
    n = self._int_to_bn(numbers.n)
    res = self._lib.RSA_set0_key(rsa_cdata, n, e, self._ffi.NULL)
    self.openssl_assert(res == 1)
    evp_pkey = self._rsa_cdata_to_evp_pkey(rsa_cdata)
    return _RSAPublicKey(self, rsa_cdata, evp_pkey)