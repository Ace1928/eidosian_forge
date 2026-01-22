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
def load_pem_public_key(self, data: bytes) -> PublicKeyTypes:
    mem_bio = self._bytes_to_bio(data)
    userdata = self._ffi.new('CRYPTOGRAPHY_PASSWORD_DATA *')
    evp_pkey = self._lib.PEM_read_bio_PUBKEY(mem_bio.bio, self._ffi.NULL, self._ffi.addressof(self._lib._original_lib, 'Cryptography_pem_password_cb'), userdata)
    if evp_pkey != self._ffi.NULL:
        evp_pkey = self._ffi.gc(evp_pkey, self._lib.EVP_PKEY_free)
        return self._evp_pkey_to_public_key(evp_pkey)
    else:
        self._consume_errors()
        res = self._lib.BIO_reset(mem_bio.bio)
        self.openssl_assert(res == 1)
        rsa_cdata = self._lib.PEM_read_bio_RSAPublicKey(mem_bio.bio, self._ffi.NULL, self._ffi.addressof(self._lib._original_lib, 'Cryptography_pem_password_cb'), userdata)
        if rsa_cdata != self._ffi.NULL:
            rsa_cdata = self._ffi.gc(rsa_cdata, self._lib.RSA_free)
            evp_pkey = self._rsa_cdata_to_evp_pkey(rsa_cdata)
            return _RSAPublicKey(self, rsa_cdata, evp_pkey)
        else:
            self._handle_key_loading_error()