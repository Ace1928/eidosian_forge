from __future__ import annotations
import ipaddress
import re
from typing import Protocol, Sequence, Union, runtime_checkable
import attr
from .exceptions import (
def verify_service_identity(cert_patterns: Sequence[CertificatePattern], obligatory_ids: Sequence[ServiceID], optional_ids: Sequence[ServiceID]) -> list[ServiceMatch]:
    """
    Verify whether *cert_patterns* are valid for *obligatory_ids* and
    *optional_ids*.

    *obligatory_ids* must be both present and match.  *optional_ids* must match
    if a pattern of the respective type is present.
    """
    if not cert_patterns:
        raise CertificateError('Certificate does not contain any `subjectAltName`s.')
    errors = []
    matches = _find_matches(cert_patterns, obligatory_ids) + _find_matches(cert_patterns, optional_ids)
    matched_ids = [match.service_id for match in matches]
    for i in obligatory_ids:
        if i not in matched_ids:
            errors.append(i.error_on_mismatch(mismatched_id=i))
    for i in optional_ids:
        if i not in matched_ids and _contains_instance_of(cert_patterns, i.pattern_class):
            errors.append(i.error_on_mismatch(mismatched_id=i))
    if errors:
        raise VerificationError(errors=errors)
    return matches