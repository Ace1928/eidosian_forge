from __future__ import annotations
import abc
import datetime
import hashlib
import ipaddress
import typing
from cryptography import utils
from cryptography.hazmat.bindings._rust import asn1
from cryptography.hazmat.bindings._rust import x509 as rust_x509
from cryptography.hazmat.primitives import constant_time, serialization
from cryptography.hazmat.primitives.asymmetric.ec import EllipticCurvePublicKey
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicKey
from cryptography.hazmat.primitives.asymmetric.types import (
from cryptography.x509.certificate_transparency import (
from cryptography.x509.general_name import (
from cryptography.x509.name import Name, RelativeDistinguishedName
from cryptography.x509.oid import (
def get_extension_for_class(self, extclass: typing.Type[ExtensionTypeVar]) -> Extension[ExtensionTypeVar]:
    if extclass is UnrecognizedExtension:
        raise TypeError("UnrecognizedExtension can't be used with get_extension_for_class because more than one instance of the class may be present.")
    for ext in self:
        if isinstance(ext.value, extclass):
            return ext
    raise ExtensionNotFound(f'No {extclass} extension was found', extclass.oid)