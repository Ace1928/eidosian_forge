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
def generate_rsa_private_key(self, public_exponent: int, key_size: int) -> rsa.RSAPrivateKey:
    rsa._verify_rsa_parameters(public_exponent, key_size)
    rsa_cdata = self._lib.RSA_new()
    self.openssl_assert(rsa_cdata != self._ffi.NULL)
    rsa_cdata = self._ffi.gc(rsa_cdata, self._lib.RSA_free)
    bn = self._int_to_bn(public_exponent)
    bn = self._ffi.gc(bn, self._lib.BN_free)
    res = self._lib.RSA_generate_key_ex(rsa_cdata, key_size, bn, self._ffi.NULL)
    self.openssl_assert(res == 1)
    evp_pkey = self._rsa_cdata_to_evp_pkey(rsa_cdata)
    return _RSAPrivateKey(self, rsa_cdata, evp_pkey, unsafe_skip_rsa_key_validation=True)