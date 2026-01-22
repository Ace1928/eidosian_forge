from __future__ import annotations
import datetime
import glob
import os
from typing import TYPE_CHECKING, Iterator
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.x509 import load_pem_x509_certificate
from kombu.utils.encoding import bytes_to_str, ensure_bytes
from celery.exceptions import SecurityError
from .utils import reraise_errors
def itercerts(self) -> Iterator[Certificate]:
    """Return certificate iterator."""
    yield from self._certs.values()