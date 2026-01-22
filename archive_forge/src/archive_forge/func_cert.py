from __future__ import annotations
import typing
from cryptography import x509
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives._serialization import PBES as PBES
from cryptography.hazmat.primitives.asymmetric import (
from cryptography.hazmat.primitives.asymmetric.types import PrivateKeyTypes
@property
def cert(self) -> typing.Optional[PKCS12Certificate]:
    return self._cert