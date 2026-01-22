from __future__ import annotations
import warnings
from typing import Sequence
from cryptography.x509 import (
from cryptography.x509.extensions import ExtensionNotFound
from pyasn1.codec.der.decoder import decode
from pyasn1.type.char import IA5String
from .exceptions import CertificateError
from .hazmat import (
def verify_certificate_ip_address(certificate: Certificate, ip_address: str) -> None:
    """
    Verify whether *certificate* is valid for *ip_address*.

    .. note::
        Nothing is verified about the *authority* of the certificate;
        the caller must verify that the certificate chains to an appropriate
        trust root themselves.

    Args:
        certificate: A *cryptography* X509 certificate object.

        ip_address:
            The IP address that *connection* should be valid for.  Can be an
            IPv4 or IPv6 address.

    Raises:
        service_identity.VerificationError:
            If *certificate* is not valid for *ip_address*.

        service_identity.CertificateError:
            If *certificate* contains invalid / unexpected data. This includes
            the case where the certificate contains no ``subjectAltName``\\ s.

    .. versionadded:: 18.1.0

    .. versionchanged:: 24.1.0
        :exc:`~service_identity.CertificateError` is raised if the certificate
        contains no ``subjectAltName``\\ s instead of
        :exc:`~service_identity.VerificationError`.
    """
    verify_service_identity(cert_patterns=extract_patterns(certificate), obligatory_ids=[IPAddress_ID(ip_address)], optional_ids=[])