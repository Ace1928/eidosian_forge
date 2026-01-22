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
def serialize_key_and_certificates_to_pkcs12(self, name: typing.Optional[bytes], key: typing.Optional[PKCS12PrivateKeyTypes], cert: typing.Optional[x509.Certificate], cas: typing.Optional[typing.List[_PKCS12CATypes]], encryption_algorithm: serialization.KeySerializationEncryption) -> bytes:
    password = None
    if name is not None:
        utils._check_bytes('name', name)
    if isinstance(encryption_algorithm, serialization.NoEncryption):
        nid_cert = -1
        nid_key = -1
        pkcs12_iter = 0
        mac_iter = 0
        mac_alg = self._ffi.NULL
    elif isinstance(encryption_algorithm, serialization.BestAvailableEncryption):
        if self._lib.CRYPTOGRAPHY_OPENSSL_300_OR_GREATER:
            nid_cert = self._lib.NID_aes_256_cbc
            nid_key = self._lib.NID_aes_256_cbc
        else:
            nid_cert = self._lib.NID_pbe_WithSHA1And3_Key_TripleDES_CBC
            nid_key = self._lib.NID_pbe_WithSHA1And3_Key_TripleDES_CBC
        pkcs12_iter = 20000
        mac_iter = 1
        mac_alg = self._ffi.NULL
        password = encryption_algorithm.password
    elif isinstance(encryption_algorithm, serialization._KeySerializationEncryption) and encryption_algorithm._format is serialization.PrivateFormat.PKCS12:
        nid_cert = 0
        nid_key = 0
        pkcs12_iter = 20000
        mac_iter = 1
        password = encryption_algorithm.password
        keycertalg = encryption_algorithm._key_cert_algorithm
        if keycertalg is PBES.PBESv1SHA1And3KeyTripleDESCBC:
            nid_cert = self._lib.NID_pbe_WithSHA1And3_Key_TripleDES_CBC
            nid_key = self._lib.NID_pbe_WithSHA1And3_Key_TripleDES_CBC
        elif keycertalg is PBES.PBESv2SHA256AndAES256CBC:
            if not self._lib.CRYPTOGRAPHY_OPENSSL_300_OR_GREATER:
                raise UnsupportedAlgorithm('PBESv2 is not supported by this version of OpenSSL')
            nid_cert = self._lib.NID_aes_256_cbc
            nid_key = self._lib.NID_aes_256_cbc
        else:
            assert keycertalg is None
        if encryption_algorithm._hmac_hash is not None:
            if not self._lib.Cryptography_HAS_PKCS12_SET_MAC:
                raise UnsupportedAlgorithm('Setting MAC algorithm is not supported by this version of OpenSSL.')
            mac_alg = self._evp_md_non_null_from_algorithm(encryption_algorithm._hmac_hash)
            self.openssl_assert(mac_alg != self._ffi.NULL)
        else:
            mac_alg = self._ffi.NULL
        if encryption_algorithm._kdf_rounds is not None:
            pkcs12_iter = encryption_algorithm._kdf_rounds
    else:
        raise ValueError('Unsupported key encryption type')
    if cas is None or len(cas) == 0:
        sk_x509 = self._ffi.NULL
    else:
        sk_x509 = self._lib.sk_X509_new_null()
        sk_x509 = self._ffi.gc(sk_x509, self._lib.sk_X509_free)
        ossl_cas = []
        for ca in cas:
            if isinstance(ca, PKCS12Certificate):
                ca_alias = ca.friendly_name
                ossl_ca = self._cert2ossl(ca.certificate)
                if ca_alias is None:
                    res = self._lib.X509_alias_set1(ossl_ca, self._ffi.NULL, -1)
                else:
                    res = self._lib.X509_alias_set1(ossl_ca, ca_alias, len(ca_alias))
                self.openssl_assert(res == 1)
            else:
                ossl_ca = self._cert2ossl(ca)
            ossl_cas.append(ossl_ca)
            res = self._lib.sk_X509_push(sk_x509, ossl_ca)
            backend.openssl_assert(res >= 1)
    with self._zeroed_null_terminated_buf(password) as password_buf:
        with self._zeroed_null_terminated_buf(name) as name_buf:
            ossl_cert = self._cert2ossl(cert) if cert else self._ffi.NULL
            ossl_pkey = self._key2ossl(key) if key is not None else self._ffi.NULL
            p12 = self._lib.PKCS12_create(password_buf, name_buf, ossl_pkey, ossl_cert, sk_x509, nid_key, nid_cert, pkcs12_iter, mac_iter, 0)
        if self._lib.Cryptography_HAS_PKCS12_SET_MAC and mac_alg != self._ffi.NULL:
            self._lib.PKCS12_set_mac(p12, password_buf, -1, self._ffi.NULL, 0, mac_iter, mac_alg)
    self.openssl_assert(p12 != self._ffi.NULL)
    p12 = self._ffi.gc(p12, self._lib.PKCS12_free)
    bio = self._create_mem_bio_gc()
    res = self._lib.i2d_PKCS12_bio(bio, p12)
    self.openssl_assert(res > 0)
    return self._read_mem_bio(bio)