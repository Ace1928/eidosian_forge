from __future__ import annotations
from contextlib import suppress
from datetime import date, datetime
from uuid import UUID
import ipaddress
import re
import typing
import warnings
from jsonschema.exceptions import FormatError
@classmethod
def cls_checks(cls, format: str, raises: _RaisesType=()) -> typing.Callable[[_F], _F]:
    warnings.warn('FormatChecker.cls_checks is deprecated. Call FormatChecker.checks on a specific FormatChecker instance instead.', DeprecationWarning, stacklevel=2)
    return cls._cls_checks(format=format, raises=raises)