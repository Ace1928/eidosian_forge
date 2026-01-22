from __future__ import annotations
from contextlib import suppress
from datetime import date, datetime
from uuid import UUID
import ipaddress
import re
import typing
import warnings
from jsonschema.exceptions import FormatError
@_checks_drafts(draft7='idn-hostname', draft201909='idn-hostname', draft202012='idn-hostname', raises=(idna.IDNAError, UnicodeError))
def is_idn_host_name(instance: object) -> bool:
    if not isinstance(instance, str):
        return True
    idna.encode(instance)
    return True