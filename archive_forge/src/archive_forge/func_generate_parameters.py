from __future__ import annotations
import abc
import typing
from cryptography.hazmat.bindings._rust import openssl as rust_openssl
from cryptography.hazmat.primitives import _serialization
def generate_parameters(generator: int, key_size: int, backend: typing.Any=None) -> DHParameters:
    from cryptography.hazmat.backends.openssl.backend import backend as ossl
    return ossl.generate_dh_parameters(generator, key_size)