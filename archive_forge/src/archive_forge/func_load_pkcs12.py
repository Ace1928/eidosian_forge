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
def load_pkcs12(self, data: bytes, password: typing.Optional[bytes]) -> PKCS12KeyAndCertificates:
    if password is not None:
        utils._check_byteslike('password', password)
    bio = self._bytes_to_bio(data)
    p12 = self._lib.d2i_PKCS12_bio(bio.bio, self._ffi.NULL)
    if p12 == self._ffi.NULL:
        self._consume_errors()
        raise ValueError('Could not deserialize PKCS12 data')
    p12 = self._ffi.gc(p12, self._lib.PKCS12_free)
    evp_pkey_ptr = self._ffi.new('EVP_PKEY **')
    x509_ptr = self._ffi.new('X509 **')
    sk_x509_ptr = self._ffi.new('Cryptography_STACK_OF_X509 **')
    with self._zeroed_null_terminated_buf(password) as password_buf:
        res = self._lib.PKCS12_parse(p12, password_buf, evp_pkey_ptr, x509_ptr, sk_x509_ptr)
    if res == 0:
        self._consume_errors()
        raise ValueError('Invalid password or PKCS12 data')
    cert = None
    key = None
    additional_certificates = []
    if evp_pkey_ptr[0] != self._ffi.NULL:
        evp_pkey = self._ffi.gc(evp_pkey_ptr[0], self._lib.EVP_PKEY_free)
        key = self._evp_pkey_to_private_key(evp_pkey, unsafe_skip_rsa_key_validation=False)
    if x509_ptr[0] != self._ffi.NULL:
        x509 = self._ffi.gc(x509_ptr[0], self._lib.X509_free)
        cert_obj = self._ossl2cert(x509)
        name = None
        maybe_name = self._lib.X509_alias_get0(x509, self._ffi.NULL)
        if maybe_name != self._ffi.NULL:
            name = self._ffi.string(maybe_name)
        cert = PKCS12Certificate(cert_obj, name)
    if sk_x509_ptr[0] != self._ffi.NULL:
        sk_x509 = self._ffi.gc(sk_x509_ptr[0], self._lib.sk_X509_free)
        num = self._lib.sk_X509_num(sk_x509_ptr[0])
        indices: typing.Iterable[int]
        if self._lib.CRYPTOGRAPHY_OPENSSL_300_OR_GREATER or self._lib.CRYPTOGRAPHY_IS_BORINGSSL:
            indices = range(num)
        else:
            indices = reversed(range(num))
        for i in indices:
            x509 = self._lib.sk_X509_value(sk_x509, i)
            self.openssl_assert(x509 != self._ffi.NULL)
            x509 = self._ffi.gc(x509, self._lib.X509_free)
            addl_cert = self._ossl2cert(x509)
            addl_name = None
            maybe_name = self._lib.X509_alias_get0(x509, self._ffi.NULL)
            if maybe_name != self._ffi.NULL:
                addl_name = self._ffi.string(maybe_name)
            additional_certificates.append(PKCS12Certificate(addl_cert, addl_name))
    return PKCS12KeyAndCertificates(key, cert, additional_certificates)