from __future__ import annotations
from contextlib import suppress
from datetime import date, datetime
from uuid import UUID
import ipaddress
import re
import typing
import warnings
from jsonschema.exceptions import FormatError
@_checks_drafts(draft7='iri-reference', draft201909='iri-reference', draft202012='iri-reference', raises=ValueError)
def is_iri_reference(instance: object) -> bool:
    if not isinstance(instance, str):
        return True
    return rfc3987.parse(instance, rule='IRI_reference')