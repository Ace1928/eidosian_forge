from __future__ import annotations
from contextlib import suppress
from datetime import date, datetime
from uuid import UUID
import ipaddress
import re
import typing
import warnings
from jsonschema.exceptions import FormatError
@_checks_drafts(draft3='time', raises=ValueError)
def is_draft3_time(instance: object) -> bool:
    if not isinstance(instance, str):
        return True
    return bool(datetime.strptime(instance, '%H:%M:%S'))