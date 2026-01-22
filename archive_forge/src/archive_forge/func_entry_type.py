from __future__ import annotations
import abc
import datetime
from cryptography import utils
from cryptography.hazmat.bindings._rust import x509 as rust_x509
from cryptography.hazmat.primitives.hashes import HashAlgorithm
@property
@abc.abstractmethod
def entry_type(self) -> LogEntryType:
    """
        Returns whether this is an SCT for a certificate or pre-certificate.
        """