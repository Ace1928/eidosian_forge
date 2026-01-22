from __future__ import annotations
import abc
import datetime
import os
import typing
from cryptography import utils
from cryptography.hazmat.bindings._rust import x509 as rust_x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import (
from cryptography.hazmat.primitives.asymmetric.types import (
from cryptography.x509.extensions import (
from cryptography.x509.name import Name, _ASN1Type
from cryptography.x509.oid import ObjectIdentifier
@property
@abc.abstractmethod
def signature_algorithm_parameters(self) -> typing.Union[None, padding.PSS, padding.PKCS1v15, ec.ECDSA]:
    """
        Returns the signature algorithm parameters.
        """