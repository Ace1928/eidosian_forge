from __future__ import annotations
import warnings
from typing import Sequence
from cryptography.x509 import (
from cryptography.x509.extensions import ExtensionNotFound
from pyasn1.codec.der.decoder import decode
from pyasn1.type.char import IA5String
from .exceptions import CertificateError
from .hazmat import (
def verify_certificate_hostname(certificate: Certificate, hostname: str) -> None:
    """
    Verify whether *certificate* is valid for *hostname*.

    .. note::
        Nothing is verified about the *authority* of the certificate;
        the caller must verify that the certificate chains to an appropriate
        trust root themselves.

    Args:
        certificate: A *cryptography* X509 certificate object.

        hostname: The hostname that *certificate* should be valid for.

    Raises:
        service_identity.VerificationError:
            If *certificate* is not valid for *hostname*.

        service_identity.CertificateError:
            If *certificate* contains invalid / unexpected data. This includes
            the case where the certificate contains no `subjectAltName`\\ s.

    .. versionchanged:: 24.1.0
        :exc:`~service_identity.CertificateError` is raised if the certificate
        contains no ``subjectAltName``\\ s instead of
        :exc:`~service_identity.VerificationError`.
    """
    verify_service_identity(cert_patterns=extract_patterns(certificate), obligatory_ids=[DNS_ID(hostname)], optional_ids=[])