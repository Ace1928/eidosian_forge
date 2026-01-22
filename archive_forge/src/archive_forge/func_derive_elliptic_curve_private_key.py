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
def derive_elliptic_curve_private_key(self, private_value: int, curve: ec.EllipticCurve) -> ec.EllipticCurvePrivateKey:
    ec_cdata = self._ec_key_new_by_curve(curve)
    group = self._lib.EC_KEY_get0_group(ec_cdata)
    self.openssl_assert(group != self._ffi.NULL)
    point = self._lib.EC_POINT_new(group)
    self.openssl_assert(point != self._ffi.NULL)
    point = self._ffi.gc(point, self._lib.EC_POINT_free)
    value = self._int_to_bn(private_value)
    value = self._ffi.gc(value, self._lib.BN_clear_free)
    with self._tmp_bn_ctx() as bn_ctx:
        res = self._lib.EC_POINT_mul(group, point, value, self._ffi.NULL, self._ffi.NULL, bn_ctx)
        self.openssl_assert(res == 1)
        bn_x = self._lib.BN_CTX_get(bn_ctx)
        bn_y = self._lib.BN_CTX_get(bn_ctx)
        res = self._lib.EC_POINT_get_affine_coordinates(group, point, bn_x, bn_y, bn_ctx)
        if res != 1:
            self._consume_errors()
            raise ValueError('Unable to derive key from private_value')
    res = self._lib.EC_KEY_set_public_key(ec_cdata, point)
    self.openssl_assert(res == 1)
    private = self._int_to_bn(private_value)
    private = self._ffi.gc(private, self._lib.BN_clear_free)
    res = self._lib.EC_KEY_set_private_key(ec_cdata, private)
    self.openssl_assert(res == 1)
    evp_pkey = self._ec_cdata_to_evp_pkey(ec_cdata)
    return _EllipticCurvePrivateKey(self, ec_cdata, evp_pkey)